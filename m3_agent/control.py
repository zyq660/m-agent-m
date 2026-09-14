# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import os
import sys
import json
import time
import openai
import argparse
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
from urllib.parse import urlparse
import mmagent.videograph
from mmagent.retrieve import search
from mmagent.local_embedding import is_local, retrieval_threshold
from transformers import AutoTokenizer
from mmagent.utils.general import load_video_graph
from mmagent.utils.chat_api import generate_messages
from mmagent.prompts import prompt_agent_verify_answer_referencing

sys.modules["videograph"] = mmagent.videograph
processing_config = json.load(open("configs/processing_config.json"))
model_name = "models/M3-Agent-Control"
gpt_model = "gpt-4o-2024-11-20"
client = None

def initialize_evaluator():
    """Only the explicit --eval-api option enables the external judge."""
    global client
    config = json.load(open("configs/api_config.json"))[gpt_model]
    client = openai.AzureOpenAI(
        azure_endpoint=config["azure_endpoint"],
        api_version=config["api_version"],
        api_key=config["api_key"],
    )

def get_response(messages, timeout=30):
    response = client.chat.completions.create(
        model=gpt_model, messages=messages, temperature=0, timeout=timeout, max_tokens=2048
    )
    return response.choices[0].message.content, response.usage.total_tokens

def get_response_with_retry(messages, timeout=30):
    for i in range(20):
        try:
            return get_response(messages, timeout)
        except Exception as e:
            time.sleep(20)
            print(f"Retry {i} times, exception: {e} from message {messages}")
            continue
    raise Exception(f"Failed to get response after 5 retries")

def eval_answer(question, predict, ground_truth):
    if predict == "":
        return False
    try:
        input = [
            {
                "type": "text",
                "content": prompt_agent_verify_answer_referencing.format(
                    question=question,
                    ground_truth_answer=ground_truth,
                    agent_answer=predict,
                ),
            }   
        ]
        messages = generate_messages(input)
        response = get_response_with_retry(messages)
        result = response[0].lower()
    except Exception as e:
        print(f"Error verifying qa: {question} | {str(e)}")
        return False
    return True if "yes" in result else False

system_prompt = "You are given a question and some relevant knowledge. Your task is to reason about whether the provided knowledge is sufficient to answer the question. If it is sufficient, output [Answer] followed by the answer. If it is not sufficient, output [Search] and generate a query that will be encoded into embeddings for a vector similarity search. The query will help retrieve additional information from a memory bank.\n\nQuestion: {question}"
instruction = f"""

Output the answer in the format:
Action: [Answer] or [Search]
Content: {{content}}

If the answer cannot be derived yet, the {{content}} should be a single search query that would help retrieve the missing information. The search {{content}} needs to be different from the previous.
You can get the mapping relationship between character ID and name by using search query such as: "What is the name of <character_{{i}}>" or "What is the character id of {{name}}".
After obtaining the mapping, it is best to use character ID instead of name for searching.
If the answer can be derived from the provided knowledge, the {{content}} is the specific answer to the question. Only name can appear in the answer, not character ID like <character_{{i}}>."""

pattern = r"Action:\s*\[(Answer|Search)\]\s*Content:\s*(.*)"

def local_generate(endpoint, prompts, max_tokens, temperature):
    if urlparse(endpoint).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Control endpoint must be local; external API fallback is disabled")
    payload = {"model": "M3-Agent-Control", "prompt": prompts,
               "max_tokens": max_tokens, "temperature": temperature,
               "top_p": 0.95, "top_k": 20, "seed": 42}
    request = Request(endpoint.rstrip("/") + "/v1/completions",
                      data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with build_opener(ProxyHandler({})).open(request, timeout=1800) as response:
        result = json.load(response)
    choices = sorted(result["choices"], key=lambda item: item["index"])
    if len(choices) != len(prompts):
        raise RuntimeError("Control returned an unexpected number of completions")
    return [item["text"] for item in choices]

def consumer(data):
    if not data["finish"]:
        started = time.monotonic()
        before_clip = data.get("before_clip", None)
        response = data["conversations"][-1]["content"]
        match_result = re.search(pattern, response.split("</think>")[-1], re.DOTALL)
        if match_result:
            action = match_result.group(1)
            content = match_result.group(2)
        else:
            action = "Search"
            content = None
        trace = {"round": len(data.get("retrieval_trace", [])) + 1,
                 "action": action, "content": content, "parsed": bool(match_result),
                 "before_clip": before_clip, "memories": {}, "clip_scores": {},
                 "model_output": response}
        if action == "Answer":
            data["response"] = content
            data["finish"] = True
        else:
            new_memories = {}
            if content:
                mem_node = load_video_graph(data["mem_path"])
                if before_clip is not None:
                    mem_node.truncate_memory_by_clip(before_clip, False)
                mem_node.refresh_equivalences()
                if "character id" in content:
                    memories, _, scores = search(mem_node, content, [], mem_wise=True, topk=20, before_clip=before_clip)
                    trace.update(search_mode="node", topk=20, threshold=0)
                    new_memories.update(memories)
                else:
                    memories, currenr_clips, scores = search(mem_node, content, data["currenr_clips"], threshold=retrieval_threshold(), topk=processing_config["topk"], before_clip=before_clip)
                    trace.update(search_mode="clip", topk=processing_config["topk"], threshold=retrieval_threshold())
                    data["currenr_clips"] = currenr_clips
                    new_memories.update(memories)
                trace["clip_scores"] = {str(key): float(value) for key, value in scores.items()}
            trace["memories"] = new_memories
            search_result = "Searched knowledge: " + json.dumps(new_memories, ensure_ascii=False).encode("utf-8", "ignore").decode("utf-8")
            if len(new_memories) == 0:
                search_result += "\n(The search result is empty. Please try searching from another perspective.)"
            data["conversations"].append({"role": "user", "content": search_result})
        trace["retrieval_seconds"] = round(time.monotonic() - started, 4)
        data.setdefault("retrieval_trace", []).append(trace)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default=("data/annotations/robot_qwen3_8b_3072.json" if is_local() else "data/annotations/robot.json"))
    parser.add_argument("--backend", choices=["local-server", "vllm"], default="local-server")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8734")
    parser.add_argument("--model", default=model_name)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--eval-api", action="store_true", help="Opt in to paid GPT-4o answer grading")
    parser.add_argument("--output", help="New JSONL file; existing files are never overwritten")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--rounds", type=int, default=processing_config["total_round"])
    parser.add_argument("--question", help="Ask a custom question about --video")
    parser.add_argument("--subset", choices=["robot", "web"], default="robot")
    parser.add_argument("--video", default="bedroom_01")
    parser.add_argument("--before-clip", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.rounds < 1 or args.limit < 0 or args.max_tokens < 1:
        parser.error("Batch size, rounds and max tokens must be positive; limit must be nonnegative")
    if args.eval_api:
        initialize_evaluator()
    dataset_name = args.data_file.split("/")[-1].split(".")[0]
    output_path = Path(args.output or os.path.join("data/results", f"{dataset_name}_control_{time.time_ns()}.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        parser.error(f"Output already exists: {output_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams
        model = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
        sampling_params = SamplingParams(temperature=args.temperature, top_p=0.95,
                                         top_k=20, max_tokens=args.max_tokens, seed=42)

    records = []
    if args.question:
        from mmagent.local_embedding import config as embedding_config
        if args.eval_api:
            parser.error("A custom question has no reference answer for --eval-api")
        records.append({"id": "custom", "question": args.question,
                        "mem_path": str(Path(embedding_config()["memory_root"]) / args.subset / f"{args.video}.pkl")})
        if args.before_clip is not None:
            records[0]["before_clip"] = args.before_clip
    else:
        datas = json.load(open(args.data_file))
        for v in datas.values():
            for qa in v["qa_list"]:
                records.append({
                "id": qa["question_id"],
                "mem_path": v["mem_path"],
                "question": qa["question"],
                "answer": qa["answer"],
                })
                if "before_clip" in qa:
                    records[-1]["before_clip"] = qa["before_clip"]
        if args.limit:
            records = records[:args.limit]
    for record in records:
        if not Path(record["mem_path"]).is_file():
            raise FileNotFoundError(record["mem_path"])
    batched_datas = [records[i:i + args.batch_size] for i in range(0, len(records), args.batch_size)]

    output_file = output_path.open("x")
    for batched_data in batched_datas:
        for i in range(len(batched_data)):
            batched_data[i]["conversations"] = [{"role": "system", "content": system_prompt.format(question=batched_data[i]["question"])}, {"role": "user", "content": "Searched knowledge: {}"}]
            batched_data[i]["finish"] = False
            batched_data[i]["currenr_clips"] = []

        for idx in range(args.rounds):
            vllm_inputs = []
            for data in batched_data:
                if data["finish"]:
                    continue
                data["conversations"][-1]["content"] += instruction
                if idx == args.rounds - 1:
                    data["conversations"][-1]["content"] += "\n(The Action of this round must be [Answer]. If there is insufficient information, you can make reasonable guesses.)"
                text = tokenizer.apply_chat_template(
                    data["conversations"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                vllm_inputs.append({"prompt_token_ids": text})

            if not vllm_inputs:
                break
            if args.backend == "local-server":
                outputs = local_generate(args.endpoint, [item["prompt_token_ids"] for item in vllm_inputs],
                                         args.max_tokens, args.temperature)
            else:
                outputs = [item.outputs[0].text for item in model.generate(
                    prompts=vllm_inputs, sampling_params=sampling_params, use_tqdm=False)]

            i = 0
            for data in batched_data:
                if data["finish"]:
                    continue
                data["conversations"].append({"role": "assistant", "content": outputs[i]})
                i += 1
            assert i == len(vllm_inputs)
            
            # Retrieval is local and CPU based; avoid forking the CUDA process.
            batched_data = [consumer(data) for data in batched_data]
            print(f"Round {idx + 1}: {sum(item['finish'] for item in batched_data)}/{len(batched_data)} answered", flush=True)

        for data in batched_data:
            if args.eval_api and "response" in data:
                data["gpt_eval"] = eval_answer(data["question"], data["response"], data["answer"])
            else:
                data["gpt_eval"] = False if args.eval_api else None
            data["evaluation"] = "gpt-4o" if args.eval_api else "disabled"
            data["status"] = "answered" if data["finish"] else "no_answer_within_round_limit"
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            output_file.flush()
            print(json.dumps({key: data.get(key) for key in ("id", "question", "response", "status")}, ensure_ascii=False), flush=True)
    output_file.close()
    print(f"Results saved to {output_path}", flush=True)
