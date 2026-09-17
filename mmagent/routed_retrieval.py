"""Five executable memory strategies with equal token caps and node-level history."""
import json
import re

from .memory_router import ACTIONS


def search_strategy(graph, query, action, seen_nodes, config, token_count,
                    before_clip=None, threshold=0.45):
    from . import retrieve as r
    from .local_embedding import encode, validate_graph
    if action not in ACTIONS:
        raise ValueError('Unknown retrieval action')
    validate_graph(graph)
    queries = r.back_translate(graph, [query]) or [query]
    # Deterministic bound; random query subsampling would confound route comparisons.
    vectors, _ = encode(queries[:100], input_type='query')
    ranked = graph.search_text_nodes(vectors, [], mode='max')
    valid = {n for n in graph.text_nodes
             if n in graph.nodes and (before_clip is None or
                                     graph.nodes[n].metadata['timestamp'] <= before_clip)}
    scores = {n: float(s) for n, s in ranked if n in valid}
    seen = set(seen_nodes)
    available = valid - seen
    entity_seeds = []
    if action == 'episodic':
        available = {n for n in available if graph.nodes[n].type == 'episodic'}
    elif action == 'semantic':
        available = {n for n in available if graph.nodes[n].type == 'semantic'}
    elif action == 'entity':
        # Use explicit graph IDs, or discover entity seeds from matching memory text.
        entity_seeds = r.get_related_nodes(graph, query)
        if not entity_seeds:
            for n, _ in ranked:
                if n in valid and scores[n] >= threshold:
                    contents = graph.nodes[n].metadata.get('contents', [])
                    for content in ([contents] if isinstance(contents, str) else contents):
                        entity_seeds.extend(r.get_related_nodes(graph, str(content)))
                    if entity_seeds:
                        break
        linked = set()
        for n in sorted(set(entity_seeds)):
            if n in graph.nodes:
                linked.update(graph.get_connected_nodes(n, type=['episodic', 'semantic']))
        available &= linked

    clip_scores = {}
    for n in sorted(available):
        clip = graph.nodes[n].metadata['timestamp']
        clip_scores[clip] = max(clip_scores.get(clip, -1), scores.get(n, -1))
    anchors = list(dict.fromkeys(int(x) for x in re.findall(r'CLIP_(\d+)', query)))
    anchors = [c for c in anchors if c in clip_scores and (before_clip is None or c <= before_clip)]
    candidates = sorted(clip_scores, key=lambda c: (-clip_scores[c], c))
    clips = list(dict.fromkeys(anchors + [c for c in candidates if clip_scores[c] >= threshold]))[:config['topk']]
    priority_nodes = []
    if action == 'temporal':
        # Reserve candidates from actual neighbors; score boosting alone cannot guarantee expansion.
        expanded = []
        for c in clips:
            for delta in range(-config.get('temporal_window', 1), config.get('temporal_window', 1) + 1):
                neighbor = c + delta
                if neighbor in graph.text_nodes_by_clip and (before_clip is None or neighbor <= before_clip):
                    expanded.append(neighbor)
        clips = list(dict.fromkeys(clips + expanded))
    for c in clips:
        nodes = [n for n in graph.text_nodes_by_clip.get(c, []) if n in available]
        nodes.sort(key=lambda n: (-scores.get(n, -1), n))
        if nodes:
            priority_nodes.append(nodes[0])
    rest = sorted((n for n in available if graph.nodes[n].metadata['timestamp'] in clips),
                  key=lambda n: (-scores.get(n, -1), n))
    ordered = list(dict.fromkeys(priority_nodes + rest))
    output, returned, oversized = {}, [], []
    selected = set()
    window = config.get('evidence_neighbor_window', 1)
    for node_id in ordered:
        if node_id in selected:
            continue
        node = graph.nodes[node_id]
        clip = node.metadata['timestamp']
        # Preserve adjacent event utterances as an atomic bundle, without changing saved memory.
        bundle = [node_id]
        if node.type == 'episodic' and window:
            events = [n for n in graph.text_nodes_by_clip[clip]
                      if n in valid and graph.nodes[n].type == 'episodic']
            index = events.index(node_id)
            bundle = events[max(0, index-window):index+window+1]
        bundle = [n for n in bundle if n not in seen and n not in selected]
        strings = []
        for n in bundle:
            strings.extend(r._memory_contents_for_node(graph, n, 'text'))
        if not strings:
            continue
        key = f'CLIP_{clip}'
        proposal = dict(output)
        proposal[key] = list(dict.fromkeys(output.get(key, []) + strings))
        size = token_count(json.dumps(proposal, ensure_ascii=False))
        if size > config['token_budget']:
            oversized.extend(bundle)
            continue
        output = proposal
        returned.extend(bundle)
        selected.update(bundle)
    details = {'action': action, 'selected_clips': clips, 'returned_node_ids': returned,
               'returned_node_count': len(returned), 'memory_tokens': token_count(json.dumps(output, ensure_ascii=False)),
               'token_budget': config['token_budget'], 'oversized_node_ids': sorted(set(oversized)),
               'entity_seeds': sorted(set(entity_seeds)), 'empty': not bool(output)}
    # A partially read clip is never excluded on future calls; only delivered nodes are visited.
    return output, list(dict.fromkeys(list(seen_nodes) + returned)), clip_scores, details
