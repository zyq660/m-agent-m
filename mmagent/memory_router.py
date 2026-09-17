"""Small CPU router. Labels/references are never part of the feature schema."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ACTIONS = ('mixed', 'episodic', 'semantic', 'temporal', 'entity')
STATE_FIELDS = ('searches', 'last_empty', 'seen_nodes', 'seen_clips',
                'last_memory_tokens', 'last_node_count', 'remaining_rounds', 'has_entity_id')
FEATURE_VERSION = 1


def load_config(path=None):
    path = Path(path) if path else REPO / 'configs/router_config.json'
    cfg = json.loads(path.read_text())
    if cfg.get('checkpoint'):
        checkpoint = Path(cfg['checkpoint'])
        cfg['checkpoint'] = str((checkpoint if checkpoint.is_absolute() else REPO / checkpoint).resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg['mode'] not in ('heuristic', 'routed_heuristic', 'fixed', 'learned'):
        raise ValueError('Unknown router mode')
    if cfg.get('scope', 'first_search') not in ('first_search', 'all_searches'):
        raise ValueError('Invalid router scope')
    for key in ('fixed_action', 'followup_action'):
        if cfg.get(key, 'mixed') not in ACTIONS:
            raise ValueError('Invalid router action: ' + key)
    if cfg.get('token_budget', 4096) < 64 or cfg.get('topk', 2) < 1:
        raise ValueError('Router budget/topk too small')
    if any(not isinstance(cfg.get(k, 1), int) or cfg.get(k, 1) < 0
           for k in ('temporal_window', 'evidence_neighbor_window')):
        raise ValueError('Neighbor windows must be nonnegative integers')
    if cfg['mode'] == 'learned' and not cfg.get('checkpoint'):
        raise ValueError('Learned routing requires a trained checkpoint; no random model fallback')


def embedding_identity():
    from .local_embedding import config
    cfg = config()
    return {k: cfg[k] for k in ('model', 'revision', 'dimensions', 'query_instruction')}


def state_features(data, query):
    import re
    history = [t for t in data.get('retrieval_trace', []) if t.get('action') == 'Search']
    last = history[-1] if history else {}
    details = last.get('router', {})
    return np.asarray([
        min(len(history), 5) / 5,
        float(bool(history) and not last.get('memories')),
        min(len(data.get('router_seen_nodes', [])), 200) / 200,
        min(len(data.get('currenr_clips', [])), 50) / 50,
        min(details.get('memory_tokens', 0), 8192) / 8192,
        min(details.get('returned_node_count', 0), 200) / 200,
        max(0, 5 - len(history)) / 5,
        float(bool(re.search(r'<(?:character|voice|person)_\d+>', query))),
    ], dtype=np.float32)


def feature_vector(question, query, state, encode=None):
    if encode is None:
        from .local_embedding import encode
    vectors, _ = encode([question, query], input_type='query')
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != 2:
        raise ValueError('Invalid router embeddings')
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    features = np.concatenate([vectors.reshape(-1), np.asarray(state, dtype=np.float32)])
    if len(state) != len(STATE_FIELDS) or not np.isfinite(features).all():
        raise ValueError('Invalid router state')
    return features


def checkpoint_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def _load_weights(path, digest):
    with np.load(path, allow_pickle=False) as archive:
        weights = {k: archive[k].copy() for k in archive.files if k != 'metadata'}
        meta = json.loads(str(archive['metadata'].item()))
    if meta['actions'] != list(ACTIONS) or meta['feature_version'] != FEATURE_VERSION:
        raise ValueError('Router checkpoint schema mismatch')
    if meta['embedding_identity'] != embedding_identity():
        raise ValueError('Router checkpoint uses a different embedding space')
    return weights, meta


def predict(features, checkpoint, config=None):
    digest = checkpoint_hash(checkpoint)
    weights, meta = _load_weights(str(checkpoint), digest)
    if config is not None:
        for key in ('scope', 'followup_action', 'token_budget', 'topk',
                    'temporal_window', 'evidence_neighbor_window'):
            if config.get(key) != meta['router_config'].get(key):
                raise ValueError('Inference differs from router training protocol: ' + key)
    x = np.asarray(features, dtype=np.float32)
    if x.shape != (weights['w0'].shape[1],):
        raise ValueError('Router feature dimension mismatch')
    for i in range(3):
        x = x @ weights[f'w{i}'].T + weights[f'b{i}']
        if i < 2:
            x = np.maximum(x, 0)
    x = np.exp(x - x.max())
    if not np.isfinite(x).all() or x.sum() <= 0:
        raise ValueError('Non-finite router predictions')
    return x / x.sum(), digest


def choose(data, query, cfg):
    """Return auditable decision; mode=heuristic is the unchanged legacy path."""
    validate_config(cfg)
    from .retrieve import infer_memory_route
    mode = cfg['mode']
    result = {'mode': mode, 'feature_version': FEATURE_VERSION, 'action': None}
    if mode == 'heuristic':
        return dict(result, heuristic_route=infer_memory_route(query))
    history = [t for t in data.get('retrieval_trace', []) if t.get('action') == 'Search']
    if cfg.get('scope', 'first_search') == 'first_search' and history:
        return dict(result, action=cfg.get('followup_action', 'mixed'), reason='fixed_followup')
    if mode == 'fixed':
        return dict(result, action=cfg.get('fixed_action', 'mixed'))
    if mode == 'routed_heuristic':
        if 'character id' in query.lower():
            action = 'entity'
        else:
            route = infer_memory_route(query)
            action = {'semantic': 'semantic', 'temporal': 'temporal', 'text': 'mixed'}.get(route, 'episodic')
        return dict(result, action=action)
    state = state_features(data, query)
    features = feature_vector(data['question'], query, state)
    probabilities, digest = predict(features, cfg['checkpoint'], config=cfg)
    return dict(result, action=ACTIONS[int(probabilities.argmax())],
                probabilities=dict(zip(ACTIONS, map(float, probabilities))),
                checkpoint_sha256=digest, state_features=state.tolist())
