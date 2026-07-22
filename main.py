import asyncio
import os
from pathlib import Path

import dotenv

from core.pipeline import MainPipeline
from core.shared import LLMFactory


# 改成你自己的后端项目绝对路径
PROJECT_PATH = "/Users/ywr/OOPS-master/sut/gramps-web-api-master"

# 自动从项目路径提取项目名
PROJECT_NAME = Path(PROJECT_PATH).name

# 改成你的大模型平台支持的模型名
MODEL_WORKER = "deepseek-v4-pro"
MODEL_PARSER = "deepseek-v4-pro"
# MODEL_WORKER = "qwen-plus"
# MODEL_PARSER = "qwen-plus"

# LLM 提供方配置；None 表示继续读取 .env 中的 LLM_API_URL/LLM_API_KEY
LLM_API_URL_OVERRIDE = None
LLM_API_KEY_OVERRIDE = None
STRUCTURED_OUTPUT_MODE = "deepseek_json"

# 所有输出都按项目名存储
LOG_BASE = f"./log/{PROJECT_NAME}"
RUN_BASE = f"./run/{PROJECT_NAME}"
STATE_BASE = f"{LOG_BASE}/states"


async def dump_after_step(pipeline, old_state, new_state):
    print(f"[OOPS][{PROJECT_NAME}] finished: {old_state} -> {new_state}")

    os.makedirs(STATE_BASE, exist_ok=True)

    try:
        pipeline.dump_states(STATE_BASE, full=False)
    except Exception as exc:
        print(f"[OOPS][{PROJECT_NAME}] dump state failed after {old_state}: {exc}")


async def main():
    dotenv.load_dotenv()

    os.makedirs(LOG_BASE, exist_ok=True)
    os.makedirs(RUN_BASE, exist_ok=True)

    oas = await MainPipeline(
        title=f"{PROJECT_NAME} API",
        version="1.0",
        project=PROJECT_PATH,
        log_base=LOG_BASE,
        upg_base=LOG_BASE,
        config=MainPipeline.Config(
            default_llm_worker_client=LLMFactory(
                MODEL_WORKER,
                api_url=LLM_API_URL_OVERRIDE,
                api_key=LLM_API_KEY_OVERRIDE,
                timeout=300,
                structured_output_mode=STRUCTURED_OUTPUT_MODE,
            ),
            default_llm_parser_client=LLMFactory(
                MODEL_PARSER,
                api_url=LLM_API_URL_OVERRIDE,
                api_key=LLM_API_KEY_OVERRIDE,
                timeout=300,
                structured_output_mode=STRUCTURED_OUTPUT_MODE,
            ),
            ignore_sufx=[
                ".class",
                ".jar",
                ".war",
                ".zip",
                ".tar",
                ".gz",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".pdf",
                ".DS_Store",
            ],
            ignore_path=[
                ".git",
                ".idea",
                ".vscode",
                "target",
                "build",
                "node_modules",
                ".venv",
                "logs",
                "log",
            ],
        ),
    ).run(on_complete=dump_after_step)

    output_path = f"{RUN_BASE}/openapi.generated.json"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(
            oas.model_dump_json(
                exclude_unset=True,
                by_alias=True,
                indent=2,
            )
        )

    print(f"[OOPS][{PROJECT_NAME}] generated: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
