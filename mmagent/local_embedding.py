"""Client for the user's local embedding process; never falls back to a cloud API."""
import json
import os
from pathlib import Path
import urllib.request
import base64
import numpy as np
from urllib.parse import urlparse

CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs/local_embedding_config.json'


def config():
    settings = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {'backend': 'api'}
    settings['backend'] = os.environ.get('M3_EMBEDDING_BACKEND', settings['backend'])
    return settings


def is_local():
    return config()['backend'] == 'local'


def request(path, body=None, timeout=600):
    endpoint = os.environ.get('M3_EMBEDDING_URL', config()['endpoint']).rstrip('/')
    if urlparse(endpoint).hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Local embedding endpoint must use the loopback interface')
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(endpoint + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    # Do not pass local text requests through the machine's external proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def encode(texts, input_type='document'):
    if not texts:
        return [], 0
    if input_type not in ('document', 'query'):
        raise ValueError('input_type must be document or query')
    result = request('/embed', {'texts': texts, 'input_type': input_type})
    settings = config()
    if result['model'] != settings['model'] or result['dimensions'] != settings['dimensions']:
        raise ValueError('The local service model/dimension does not match the project configuration')
    if (result['metadata']['revision'] != settings['revision'] or
            result['metadata']['query_instruction'] != settings['query_instruction']):
        raise ValueError('Local service revision/query instruction does not match the configuration')
    vectors = np.frombuffer(base64.b64decode(result['vectors_f32_base64']), dtype='<f4')
    vectors = vectors.reshape(result['count'], result['dimensions'])
    if len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError('Invalid local embedding response')
    return vectors.tolist(), result['tokens']


def validate_graph(graph):
    if not is_local():
        meta = getattr(graph, 'text_embedding_metadata', {})
        if meta.get('model', '').startswith('Qwen/'):
            raise ValueError('Qwen memory graphs require the local Qwen query encoder; use an original graph with API mode')
        return
    settings = config()
    meta = getattr(graph, 'text_embedding_metadata', {})
    if (meta.get('model') != settings['model'] or
            meta.get('revision') != settings['revision'] or
            meta.get('dimensions') != settings['dimensions'] or
            meta.get('query_instruction') != settings['query_instruction']):
        raise ValueError('This graph has not been converted to the configured Qwen embedding space. '
                         'Use data/annotations/robot_qwen3_8b_3072.json or '
                         'web_qwen3_8b_3072.json, and the matching new memory graphs.')


def retrieval_threshold(original=0.5):
    return config()['retrieval_threshold'] if is_local() else original


def prepare_graph_for_text_updates(graph):
    """Mark empty new graphs and reject adding incompatible vectors to existing graphs."""
    if not is_local() or any(node.type in ('episodic','semantic') for node in graph.nodes.values()):
        validate_graph(graph)
        return
    identity = request('/health')
    graph.text_embedding_metadata = {k:identity[k] for k in (
        'model','revision','dimensions','dtype','pooling','normalization','query_instruction')}
    validate_graph(graph)
