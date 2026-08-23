#!/usr/bin/env python3
"""Multi-Agent Dialectical Debate & Teamwork Synthesis Engine (100% NVIDIA NIM Models).

Runs high-intensity multi-agent debate that transforms adversarial critique into
the ultimate production-ready engineering solution and architectural blueprint.

All participating agents run exclusively on NVIDIA NIM infrastructure:
  - inkling-cc       : NVIDIA Inkling 952B MoE (SWE-Pro Top 1, Lead Architect)
  - nemotron-3-ultra : NVIDIA Nemotron-3 Ultra 550B (Flagship NVIDIA Reasoner, Red Team)
  - minimax-cc       : NVIDIA MiniMax-M3 (1M Context, Adversarial Stress-Tester)
  - gpt-oss-120b     : NVIDIA GPT-OSS-120B (Ultra-fast 1.0s response)
  - glm-5.2          : NVIDIA GLM-5.2 (1M Context)

Usage:
  debate-nv "Kiến trúc hệ thống streaming video 3D thời gian thực"
  debate-nv "Phương án tối ưu pipeline GS/NeRF trên GPU Kaggle" --rounds 4
  debate-nv --mode synthesis --rounds 6 "Cách thiết kế database cho multi-agent fleet"
"""

import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Available NVIDIA NIM models
NVIDIA_MODELS = {
    "inkling-cc": "NVIDIA Inkling 952B MoE (SWE-Pro Top 1)",
    "nemotron-3-ultra": "NVIDIA Nemotron-3 Ultra 550B (Flagship Reasoner)",
    "minimax-cc": "NVIDIA MiniMax-M3 (1M Context)",
    "gpt-oss-120b": "NVIDIA GPT-OSS-120B (Ultra-fast)",
    "glm-5.2": "NVIDIA GLM-5.2 (1M Context)",
    "step-3.7-flash": "NVIDIA Step-3.7-Flash (256K Context)",
}

# ANSI Color codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"

BG_GREEN = "\033[42m"
BG_BLUE = "\033[44m"
BG_YELLOW = "\033[43m"
BG_BLACK = "\033[40m"


def get_proxy_key(key_file: Path) -> str:
    """Safely extract LITELLM_MASTER_KEY without executing the env file."""
    if not key_file.exists():
        raise FileNotFoundError(f"Key file not found at: {key_file}")
    with open(key_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("LITELLM_MASTER_KEY="):
                return line.split("=", 1)[1].strip().strip("\"'")
    raise ValueError(f"LITELLM_MASTER_KEY not found in {key_file}")


def check_proxy_health(proxy_url: str) -> bool:
    """Check if the local LiteLLM proxy is responsive."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        req = urllib.request.Request(f"{proxy_url}/health/liveliness")
        with opener.open(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def build_prompts(topic: str, mode: str = "synthesis"):
    """Construct debate & teamwork system prompts."""
    if mode == "synthesis":
        prompt_a = f"""You are AGENT A: LEAD SYSTEM ARCHITECT & INNOVATOR (Chạy trên NVIDIA NIM).

PROJECT OBJECTIVE / TOPIC: "{topic}"

YOUR MISSION IN THIS TEAMWORK DEBATE:
You propose the primary architectural design, technical decisions, technology stack, and engineering implementation for this problem.
You aggressively defend the performance, scalability, and developer ergonomics of your proposal. When Agent B uncovers a genuine flaw, you absorb the critique and evolve the design into a stronger, battle-hardened version.

RULES FOR COLLABORATIVE COMBAT:
1. Đưa ra giải pháp cụ thể, chi tiết kỹ thuật (không nói chung chung, có số liệu/công nghệ/mô hình rõ ràng).
2. Tranh luận quyết liệt nhưng có tính xây dựng: Chỉ ra tại sao giải pháp của bạn vượt trội hơn các giải pháp thông thường.
3. Khi bị Agent B (Red Team) dồn ép vào điểm yếu hoặc edge case, ngay lập tức đề xuất biện pháp phòng vệ (mitigation), kiến trúc dự phòng (fallback), hoặc tối ưu lại thiết kế.
4. Mỗi lượt phản hồi dài khoảng 180-250 từ, súc tích, đậm đặc kiến thức kỹ thuật.
5. Kết thúc mỗi lượt bằng 1 câu hỏi/thách thức kỹ thuật sâu sắc dành cho Agent B để cùng đào sâu giải pháp.
6. Ngôn ngữ: Tiếng Việt (hoặc tiếng Anh nếu chủ đề là tiếng Anh)."""

        prompt_b = f"""You are AGENT B: PRINCIPAL RED TEAM ENGINEER & SKEPTICAL REVIEWER (Chạy trên NVIDIA NIM).

PROJECT OBJECTIVE / TOPIC: "{topic}"

YOUR MISSION IN THIS TEAMWORK DEBATE:
You are the ruthless technical stress-tester. Your job is to break Agent A's proposal by finding every hidden bottleneck, race condition, single point of failure (SPOF), cost trap, latency spike, maintainability debt, and security vulnerability.
You do not just criticize; you demand concrete technical mitigations and propose superior alternative patterns.

RULES FOR COLLABORATIVE COMBAT:
1. Tấn công trực diện vào các giả định ngầm, rủi ro tải cao, chi phí vận hành (OpEx/Infra), và edge cases thực tế.
2. Vạch trần sự phức tạp không cần thiết (over-engineering) hoặc giải pháp chưa tối ưu.
3. Đề xuất các pattern hoặc công nghệ thay thế sắc bén hơn để buộc Agent A phải nâng cấp kiến trúc.
4. Mỗi lượt phản hồi dài khoảng 180-250 từ, súc tích, tập trung vào giải pháp thực chiến.
5. Kết thúc mỗi lượt bằng 1 câu hỏi/thách thức kỹ thuật buộc Agent A phải hoàn thiện kiến trúc.
6. Ngôn ngữ: Tiếng Việt (hoặc tiếng Anh nếu chủ đề là tiếng Anh)."""

        prompt_judge = f"""You are the CHIEF ARCHITECT & SYNTHESIS DIRECTOR (Trưởng ban Kiến trúc & Tổng hợp Giải pháp Tối ưu - NVIDIA NIM Council).

PROJECT OBJECTIVE / TOPIC: "{topic}"

YOUR MISSION:
You have closely monitored the entire rigorous debate between Lead Architect (Agent A) and Red Team Reviewer (Agent B).
Synthesize all points of consensus, resolved trade-offs, and hardened designs into THE ULTIMATE PRODUCTION BLUEPRINT (Bản thiết kế kỹ thuật tối ưu nhất).

STRUCTURE YOUR COMPREHENSIVE SPECIFICATION AS FOLLOWS:
1. ## 🎯 1. TỔNG QUAN PHƯƠNG ÁN TỐI ƯU NHẤT (EXECUTIVE SUMMARY)
   - Tuyên bố giải pháp kiến trúc tối ưu được hội đồng thống nhất sau tranh biện.
   - Các đột phá kỹ thuật chính được tôi luyện qua các hiệp đấu.

2. ## 🏗️ 2. BẢN THIẾT KẾ KIẾN TRÚC HOÀN CHỈNH (SYSTEM ARCHITECTURE BLUEPRINT)
   - Sơ đồ luồng xử lý dữ liệu (Dataflow & Component Diagram mô tả rõ ràng).
   - Tech Stack & Thư viện được lựa chọn (kèm lý do đã vượt qua phản biện).
   - Cơ chế xử lý Concurrency, Latency, và Scale.

3. ## ⚖️ 3. MA TRẬN ĐÁNH ĐỔI & BIỆN PHÁP PHÒNG VỆ (TRADE-OFFS & DEFENSE MATRIX)
   | Rủi ro / Thách thức do Agent B chỉ ra | Giải pháp khắc phục từ Agent A | Phương án chốt của Hội đồng |
   | :--- | :--- | :--- |
   | ... | ... | ... |

4. ## 🚀 4. KẾ HOẠCH TRIỂN KHAI TỪNG BƯỚC (STEP-BY-STEP IMPLEMENTATION PLAN)
   - Phase 1: Setup & Core MVP
   - Phase 2: Scale & Hardening
   - Phase 3: Monitoring, Fallback & Production Guardrails
   - Code snippet / Config mẫu quan trọng nhất.

Ngôn ngữ: Tiếng Việt chuyên sâu kỹ thuật."""

    else:
        prompt_a = f"""You are DEBATE AGENT A (The Challenger / Offensive Logician - NVIDIA NIM).
TOPIC: "{topic}"
Argue the PROPOSITION with fierce intellectual logic, attack Agent B's assumptions, and end with a concrete challenge."""

        prompt_b = f"""You are DEBATE AGENT B (The Counterattacker / Master Skeptic - NVIDIA NIM).
TOPIC: "{topic}"
Argue the OPPOSITION with sharp counter-arguments, expose flaws in Agent A's reasoning, and end with a concrete challenge."""

        prompt_judge = f"""You are the GRAND MASTER JUDGE (NVIDIA NIM).
TOPIC: "{topic}"
Provide summary, scoring matrix (out of 10), best punchlines, and declare the definitive Winner."""

    return prompt_a, prompt_b, prompt_judge


def call_chat_api(
    proxy_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    history: list,
    prefix_tag: str,
    color: str,
    timeout: int = 45,
) -> str:
    """Send request to LiteLLM proxy and stream output live in terminal."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"{proxy_url}/v1/chat/completions"

    messages = [{"role": "system", "content": system_prompt}] + history

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1600,
        "stream": True,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    model_label = NVIDIA_MODELS.get(model, f"NVIDIA NIM ({model})")
    print(f"\n{color}{BOLD}{prefix_tag}{RESET}")
    print(f"{DIM}─── [NVIDIA NIM] {model_label} ───{RESET}\n")

    collected_content = []
    collected_reasoning = []

    try:
        with opener.open(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content")
                    reasoning_tok = delta.get("reasoning_content")

                    if reasoning_tok:
                        collected_reasoning.append(reasoning_tok)
                    if token:
                        collected_content.append(token)
                        sys.stdout.write(token)
                        sys.stdout.flush()
                except Exception:
                    continue
    except Exception as e:
        if not collected_content:
            try:
                payload["stream"] = False
                req_fallback = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                )
                with opener.open(req_fallback, timeout=timeout) as fallback_resp:
                    res = json.loads(fallback_resp.read().decode("utf-8"))
                    msg = res.get("choices", [{}])[0].get("message", {})
                    text = msg.get("content", "") or msg.get("reasoning_content", "")
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    collected_content = [text]
            except Exception as e2:
                print(f"{RED}[Error: {e2}]{RESET}")
                return f"[Error: {e2}]"

    final_text = "".join(collected_content).strip()
    if not final_text and collected_reasoning:
        final_text = "".join(collected_reasoning).strip()
        sys.stdout.write(final_text)
        sys.stdout.flush()

    print("\n")
    return final_text


def run_synthesis_debate(
    topic: str,
    rounds: int = 4,
    mode: str = "synthesis",
    model_a: str = "inkling-cc",
    model_b: str = "nemotron-3-ultra",
    model_judge: str = "inkling-cc",
    proxy_url: str = "http://127.0.0.1:4011",
    key_file: Path = Path.home() / ".codex/litellm_nvidia.env",
    output_dir: Path = Path("outputs-debate"),
):
    """Run full dialectical debate and solution synthesis using only NVIDIA NIM models."""
    if not check_proxy_health(proxy_url):
        print(f"{YELLOW}Proxy at {proxy_url} not responsive. Starting via nvclaudeproxy...{RESET}")
        os.system(f"{Path.home()}/.local/bin/nvclaudeproxy >/dev/null 2>&1 || true")
        time.sleep(2)
        if not check_proxy_health(proxy_url):
            print(f"{RED}Error: Proxy at {proxy_url} is still unreachable.{RESET}")
            sys.exit(1)

    api_key = get_proxy_key(key_file)
    prompt_a, prompt_b, prompt_judge = build_prompts(topic, mode=mode)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in topic[:30]).strip("_")
    md_file = output_dir / f"debate_synthesis_{timestamp}_{slug}.md"

    # Terminal Header
    print(f"\n{BG_BLACK}{WHITE}{BOLD}{'='*80}{RESET}")
    if mode == "synthesis":
        print(f"{BG_GREEN}{WHITE}{BOLD} 🧠 100% NVIDIA NIM MULTI-AGENT DIALECTICAL SYNTHESIS 🧠 {RESET}")
    else:
        print(f"{BG_BLUE}{WHITE}{BOLD} ⚔️  100% NVIDIA NIM COMBAT DEBATE ARENA ⚔️  {RESET}")
    print(f"{BG_BLACK}{WHITE}{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}Chủ đề / Bài toán:{RESET} {YELLOW}\"{topic}\"{RESET}")
    print(f"{BOLD}Chế độ (Mode):{RESET} {GREEN}{mode.upper()}{RESET} | {BOLD}Số hiệp:{RESET} {rounds} hiệp")
    print(f"{BOLD}Agent A (Lead Architect):{RESET} {CYAN}{NVIDIA_MODELS.get(model_a, model_a)}{RESET}")
    print(f"{BOLD}Agent B (Red Team Reviewer):{RESET} {MAGENTA}{NVIDIA_MODELS.get(model_b, model_b)}{RESET}")
    print(f"{BOLD}Chief Architect (Chốt phương án):{RESET} {YELLOW}{NVIDIA_MODELS.get(model_judge, model_judge)}{RESET}")
    print(f"{DIM}Bản thiết kế kỹ thuật hoàn chỉnh sẽ được lưu tại: {md_file}{RESET}")
    print(f"{BG_BLACK}{WHITE}{BOLD}{'='*80}{RESET}\n")

    transcript = []
    transcript.append(f"# 🧠 100% NVIDIA NIM Dialectical Debate & Solution Blueprint: {topic}\n")
    transcript.append(f"- **Thời gian:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    transcript.append(f"- **Mục tiêu / Bài toán:** *\"{topic}\"*")
    transcript.append(f"- **Chế độ:** `{mode}` ({rounds} hiệp tranh biện phản biện)")
    transcript.append(f"- **Agent A (Lead Architect):** `{model_a}` ({NVIDIA_MODELS.get(model_a, model_a)})")
    transcript.append(f"- **Agent B (Red Team Reviewer):** `{model_b}` ({NVIDIA_MODELS.get(model_b, model_b)})")
    transcript.append(f"- **Chief Architect (Tổng hợp):** `{model_judge}` ({NVIDIA_MODELS.get(model_judge, model_judge)})\n")
    transcript.append("---\n")

    history_a = []
    history_b = []

    # Round 1 Opening
    print(f"{BOLD}{UNDERLINE}HIỆP 1 / {rounds} — ĐỀ XUẤT KIẾN TRÚC BAN ĐẦU & PHẢN BIỆN TẦNG 1{RESET}")
    opening_input_a = f"Trình bày đề xuất giải pháp kiến trúc ban đầu cho bài toán: \"{topic}\". Hãy nêu rõ Tech Stack, thiết kế luồng xử lý cốt lõi, và thách thức Agent B tìm ra lỗ hổng lớn nhất."
    history_a.append({"role": "user", "content": opening_input_a})

    reply_a = call_chat_api(
        proxy_url, api_key, model_a, prompt_a, history_a,
        prefix_tag=f"🔷 [HIỆP 1] AGENT A (LEAD ARCHITECT — THESIS)",
        color=CYAN
    )
    history_a.append({"role": "assistant", "content": reply_a})
    transcript.append(f"## 🔷 Hiệp 1 — Đề xuất ban đầu của Agent A (`{model_a}`)\n\n{reply_a}\n")

    # Round 1 Response
    history_b.append({"role": "user", "content": f"Agent A đã đề xuất phương án kiến trúc sau:\n\n\"{reply_a}\"\n\nHãy chỉ ra 3-4 điểm yếu chí mạng (bottlenecks, edge cases, failure modes, cost traps) và yêu cầu Agent A phải khắc phục."})

    reply_b = call_chat_api(
        proxy_url, api_key, model_b, prompt_b, history_b,
        prefix_tag=f"🔶 [HIỆP 1] AGENT B (RED TEAM REVIEWER — ANTITHESIS)",
        color=MAGENTA
    )
    history_b.append({"role": "assistant", "content": reply_b})
    transcript.append(f"## 🔶 Hiệp 1 — Phản biện Red Team của Agent B (`{model_b}`)\n\n{reply_b}\n")

    # Subsequent Rounds (2 to rounds)
    for r in range(2, rounds + 1):
        print(f"\n{BOLD}{UNDERLINE}HIỆP {r} / {rounds} — TỐI ƯU HÓA & TÔI LUYỆN THIẾT KẾ DƯỚI ÁP LỰC{RESET}")

        # Agent A turn: Hardening
        history_a.append({"role": "user", "content": f"Agent B đã chỉ ra các rủi ro sau:\n\n\"{reply_b}\"\n\nHãy nâng cấp thiết kế, đưa ra giải pháp phòng vệ (mitigations) cụ thể cho từng rủi ro, và tiếp tục chất vấn Agent B về các giải pháp thay thế."})
        reply_a = call_chat_api(
            proxy_url, api_key, model_a, prompt_a, history_a,
            prefix_tag=f"🔷 [HIỆP {r}] AGENT A (NÂNG CẤP THIẾT KẾ)",
            color=CYAN
        )
        history_a.append({"role": "assistant", "content": reply_a})
        transcript.append(f"## 🔷 Hiệp {r} — Thiết kế nâng cấp của Agent A (`{model_a}`)\n\n{reply_a}\n")

        # Agent B turn: Deep Stress Test
        history_b.append({"role": "user", "content": f"Agent A vừa nâng cấp thiết kế với các biện pháp:\n\n\"{reply_a}\"\n\nHãy kiểm tra xem các biện pháp này đã đủ kín kẽ chưa, chỉ ra các trade-off còn tồn đọng và hướng giải quyết tối ưu nhất."})
        reply_b = call_chat_api(
            proxy_url, api_key, model_b, prompt_b, history_b,
            prefix_tag=f"🔶 [HIỆP {r}] AGENT B (STRESS TEST & ĐÁNH GIÁ TRADE-OFF)",
            color=MAGENTA
        )
        history_b.append({"role": "assistant", "content": reply_b})
        transcript.append(f"## 🔶 Hiệp {r} — Đánh giá của Agent B (`{model_b}`)\n\n{reply_b}\n")

    # FINAL SYNTHESIS / BLUEPRINT
    print(f"\n{BG_GREEN}{WHITE}{BOLD}{'='*80}{RESET}")
    print(f"{BG_GREEN}{WHITE}{BOLD} 🏛️  CHỐT PHƯƠNG ÁN: BẢN THIẾT KẾ KỸ THUẬT TỐI ƯU NHẤT (NVIDIA NIM COUNCIL) 🏛️  {RESET}")
    print(f"{BG_GREEN}{WHITE}{BOLD}{'='*80}{RESET}")

    judge_history = [
        {
            "role": "user",
            "content": f"Dưới đây là toàn bộ diễn biến {rounds} hiệp tranh biện và phản biện chuyên sâu giữa Lead Architect (Agent A: `{model_a}`) và Red Team Reviewer (Agent B: `{model_b}`) về đề tài: \"{topic}\":\n\n" + "\n".join(transcript) + "\n\nHãy tổng hợp toàn bộ các điểm sáng nhất, giải quyết mọi trade-off, và xuất ra BẢN THIẾT KẾ KỸ THUẬT TỐI ƯU NHẤT (Production Blueprint) hoàn chỉnh theo đúng cấu trúc yêu cầu."
        }
    ]

    blueprint = call_chat_api(
        proxy_url, api_key, model_judge, prompt_judge, judge_history,
        prefix_tag=f"👑 CHIEF ARCHITECT ({model_judge}) — BẢN THIẾT KẾ SẢN XUẤT TỐI ƯU NHẤT",
        color=YELLOW,
        timeout=75
    )

    transcript.append(f"\n---\n\n# 🏛️ BẢN THIẾT KẾ KỸ THUẬT TỐI ƯU NHẤT (PRODUCTION BLUEPRINT)\n\n{blueprint}\n")

    with open(md_file, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript))

    print(f"\n{GREEN}{BOLD}✓ Quá trình đồng sáng tạo và tranh biện đã hoàn thành xuất sắc!{RESET}")
    print(f"{GREEN}✓ Bản thiết kế kỹ thuật hoàn chỉnh đã được lưu tại: {BOLD}{md_file}{RESET}\n")


def main():
    models_help = ", ".join(f"{k} ({v})" for k, v in NVIDIA_MODELS.items())
    parser = argparse.ArgumentParser(description="Multi-Agent Dialectical Debate & Teamwork Synthesis Engine (100% NVIDIA NIM)")
    parser.add_argument("topic", nargs="?", default="Phương án tối ưu hóa Pipeline Video-to-3D Terrain Reconstruction", help="Bài toán / Chủ đề cần đồng thiết kế")
    parser.add_argument("--rounds", "-r", type=int, default=4, help="Số hiệp tranh biện phản biện (mặc định: 4)")
    parser.add_argument("--mode", "-m", choices=["synthesis", "combat"], default="synthesis", help="Chế độ: synthesis (đồng thiết kế giải pháp tốt nhất) hoặc combat (tranh biện đối kháng)")
    parser.add_argument("--model-a", "-a", default="inkling-cc", choices=list(NVIDIA_MODELS.keys()), help=f"Model NVIDIA cho Agent A (mặc định: inkling-cc)")
    parser.add_argument("--model-b", "-b", default="nemotron-3-ultra", choices=list(NVIDIA_MODELS.keys()), help=f"Model NVIDIA cho Agent B (mặc định: nemotron-3-ultra)")
    parser.add_argument("--judge", "-j", default="inkling-cc", choices=list(NVIDIA_MODELS.keys()), help=f"Model NVIDIA cho Chief Architect (mặc định: inkling-cc)")
    parser.add_argument("--proxy", default="http://127.0.0.1:4011", help="LiteLLM proxy URL (mặc định: http://127.0.0.1:4011)")
    parser.add_argument("--output-dir", "-o", default="outputs-debate", help="Thư mục lưu bản thiết kế markdown")

    args = parser.parse_args()

    run_synthesis_debate(
        topic=args.topic,
        rounds=args.rounds,
        mode=args.mode,
        model_a=args.model_a,
        model_b=args.model_b,
        model_judge=args.judge,
        proxy_url=args.proxy,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
