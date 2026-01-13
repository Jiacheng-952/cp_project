# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import base64
import json
import logging
import os
from typing import Annotated, List, cast
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage
from langgraph.types import Command

from src.config.report_style import ReportStyle
from src.config.tools import SELECTED_RAG_PROVIDER

# Legacy graph import removed - server uses specialized graphs for different features
from src.llms.llm import get_configured_llm_models
from src.podcast.graph.builder import build_graph as build_podcast_graph
from src.ppt.graph.builder import build_graph as build_ppt_graph
from src.prompt_enhancer.graph.builder import build_graph as build_prompt_enhancer_graph
from src.prose.graph.builder import build_graph as build_prose_graph
from src.rag.builder import build_retriever
from src.rag.retriever import Resource

# Import for chat functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from src.server.chat_request import (
    ChatRequest,
    EnhancePromptRequest,
    GeneratePodcastRequest,
    GeneratePPTRequest,
    GenerateProseRequest,
    TTSRequest,
)
from src.server.config_request import ConfigResponse
from src.server.mcp_request import MCPServerMetadataRequest, MCPServerMetadataResponse
from src.server.mcp_utils import load_mcp_tools
from src.server.rag_request import (
    RAGConfigResponse,
    RAGResourceRequest,
    RAGResourcesResponse,
)
from src.tools import VolcengineTTS

logger = logging.getLogger(__name__)

INTERNAL_SERVER_ERROR_DETAIL = "Internal Server Error"

app = FastAPI(
    title="DeerFlow API",
    description="API for Deer",
    version="0.1.0",
)

# Add CORS middleware
# It's recommended to load the allowed origins from an environment variable
# for better security and flexibility across different environments.
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:3001,http://127.0.0.1:8000,http://[::1]:3000,http://[::1]:3001,http://[::1]:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

# 如果环境变量只设置了部分地址，确保包含IPv6地址和端口3001、3002
if "http://[::1]:3000" not in allowed_origins:
    allowed_origins.extend(["http://[::1]:3000", "http://[::1]:3001", "http://[::1]:3002", "http://[::1]:8000"])

# 确保包含端口3001和3002（前端服务器默认端口）
if "http://localhost:3001" not in allowed_origins:
    allowed_origins.append("http://localhost:3001")
if "http://127.0.0.1:3001" not in allowed_origins:
    allowed_origins.append("http://127.0.0.1:3001")
if "http://localhost:3002" not in allowed_origins:
    allowed_origins.append("http://localhost:3002")
if "http://127.0.0.1:3002" not in allowed_origins:
    allowed_origins.append("http://127.0.0.1:3002")

logger.info(f"Allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Restrict to specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE", "PATCH"],  # Allow common HTTP methods
    allow_headers=["*"],  # Now allow all headers, but can be restricted further
)


@app.get("/")
async def root():
    """根路径路由，返回服务器状态信息"""
    return {
        "message": "DeerFlow API Server is running",
        "version": "0.1.0",
        "available_endpoints": [
            "POST /api/chat/stream",
            "POST /api/tts", 
            "POST /api/podcast/generate",
            "POST /api/ppt/generate",
            "POST /api/prose/generate",
            "POST /api/prompt/enhance",
            "GET /api/config",
            "GET /api/rag/config",
            "GET /api/rag/resources",
            "POST /api/mcp/server/metadata"
        ]
    }

# Chat graph will be created on-demand for chat endpoints
def create_chat_graph():
    """Create an OptAgent workflow graph for optimization problem solving."""
    from src.graph.builder import build_optag_graph
    
    # Build the complete OptAgent workflow graph
    try:
        optag_graph = build_optag_graph()
        return optag_graph
    except Exception as e:
        raise RuntimeError(f"Failed to build OptAgent graph: {e}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    # Check if MCP server configuration is enabled
    mcp_enabled = os.getenv("ENABLE_MCP_SERVER_CONFIGURATION", "false").lower() in [
        "true",
        "1",
        "yes",
    ]

    # Validate MCP settings if provided
    if request.mcp_settings and not mcp_enabled:
        raise HTTPException(
            status_code=403,
            detail="MCP server configuration is disabled. Set ENABLE_MCP_SERVER_CONFIGURATION=true to enable MCP features.",
        )

    thread_id = request.thread_id
    if thread_id == "__default__":
        thread_id = str(uuid4())
    return StreamingResponse(
        _astream_workflow_generator(
            request.model_dump()["messages"],
            thread_id,
            request.resources,
            request.max_plan_iterations,
            request.max_step_num,
            request.max_search_results,
            request.auto_accepted_plan,
            request.interrupt_feedback,
            request.mcp_settings if mcp_enabled else {},
            request.enable_background_investigation,
            request.report_style,
            request.enable_deep_thinking,
        ),
        media_type="text/event-stream",
    )


async def _astream_workflow_generator(
    messages: List[dict],
    thread_id: str,
    resources: List[Resource],
    max_plan_iterations: int,
    max_step_num: int,
    max_search_results: int,
    auto_accepted_plan: bool,
    interrupt_feedback: str,
    mcp_settings: dict,
    enable_background_investigation: bool,
    report_style: ReportStyle,
    enable_deep_thinking: bool,
):
    # Create OptAgent graph for this request
    optag_graph = create_chat_graph()
    
    # Extract the last message content for optimization problem
    last_message_content = messages[-1]["content"] if messages else ""
    
    # Prepare input for the OptAgent graph
    input_state = {
        "messages": messages,
        "problem_statement": last_message_content,
        "max_corrections": 5,
        "debug_mode": False
    }
    
    # Track the workflow progress
    step_count = 0
    
    async for event_data in optag_graph.astream(
        input_state,
        stream_mode="values",
    ):
        # Handle different types of events from OptAgent workflow
        if isinstance(event_data, dict):
            # This is a state update from the workflow
            # Determine current node based on the presence of specific fields
            current_node = "unknown"
            
            # Check for specific fields that indicate which node is active
            # Use more specific checks to avoid false positives
            if "final_solution" in event_data:
                current_node = "reporter"
            elif "correction_count" in event_data and event_data.get("correction_count", 0) > 0:
                current_node = "corrector"
            elif "verification_result" in event_data or "verification_passed" in event_data:
                current_node = "verifier"
            elif "current_model" in event_data or "current_code" in event_data:
                current_node = "modeler"
            elif "classification_result" in event_data:
                current_node = "classifier"
            elif "visualization_data" in event_data:
                current_node = "visualizer"
            
            # Generate appropriate event based on current workflow node
            if current_node == "classifier":
                content = "🔍 正在分析问题类型..."
                if event_data.get("classification_result"):
                    classification_result = event_data.get("classification_result", {})
                    content = f"✅ 问题分类完成！\n\n问题类型: {classification_result.get('problem_type', '未知')}\n推荐求解器: {classification_result.get('solver_type', '未知')}"
            elif current_node == "modeler":
                content = "🔧 正在构建优化模型..."
                if event_data.get("current_code"):
                    content = f"✅ 模型构建完成！\n\n模型代码:\n```python\n{event_data.get('current_code', '')}\n```"
            elif current_node == "verifier":
                content = "🔍 正在验证模型和代码..."
                if event_data.get("verification_result"):
                    content = f"🔍 验证结果:\n{event_data.get('verification_result', '')}"
            elif current_node == "corrector":
                content = "🛠️ 正在修正模型问题..."
                correction_count = event_data.get("correction_count", 0)
                content = f"🛠️ 第{correction_count}次修正中..."
            elif current_node == "visualizer":
                content = "📊 正在生成可视化结果..."
                if event_data.get("visualization_data"):
                    content = "✅ 可视化结果生成完成！"
            elif current_node == "reporter":
                content = "� 正在生成最终报告..."
                if event_data.get("final_solution"):
                    final_solution = event_data.get("final_solution", {})
                    content = f"🎉 优化完成！\n\n最终报告:\n{final_solution.get('final_report', '')}"
            else:
                content = f"🔄 工作流执行中... ({current_node})"
            
            event_stream_message: dict[str, any] = {
                "thread_id": thread_id,
                "agent": "optagent",
                "id": str(uuid4()),
                "role": "assistant",
                "content": content,
                "workflow_node": current_node,
                "step_count": step_count
            }
            
            # Add specific workflow data if available
            if event_data.get("verification_passed") is not None:
                event_stream_message["verification_passed"] = event_data.get("verification_passed")
            if event_data.get("correction_count") is not None:
                event_stream_message["correction_count"] = event_data.get("correction_count")
            # Add final solution data if available
            if event_data.get("final_solution") is not None:
                event_stream_message["final_solution"] = event_data.get("final_solution")
            
            step_count += 1
            
            # Yield the workflow progress event
            yield _make_event("workflow_progress", event_stream_message)


def _make_event(event_type: str, data: dict[str, any]):
    if data.get("content") == "":
        data.pop("content")
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/tts")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using volcengine TTS API."""
    app_id = os.getenv("VOLCENGINE_TTS_APPID", "")
    if not app_id:
        raise HTTPException(status_code=400, detail="VOLCENGINE_TTS_APPID is not set")
    access_token = os.getenv("VOLCENGINE_TTS_ACCESS_TOKEN", "")
    if not access_token:
        raise HTTPException(
            status_code=400, detail="VOLCENGINE_TTS_ACCESS_TOKEN is not set"
        )

    try:
        cluster = os.getenv("VOLCENGINE_TTS_CLUSTER", "volcano_tts")
        voice_type = os.getenv("VOLCENGINE_TTS_VOICE_TYPE", "BV700_V2_streaming")

        tts_client = VolcengineTTS(
            appid=app_id,
            access_token=access_token,
            cluster=cluster,
            voice_type=voice_type,
        )
        # Call the TTS API
        result = tts_client.text_to_speech(
            text=request.text[:1024],
            encoding=request.encoding,
            speed_ratio=request.speed_ratio,
            volume_ratio=request.volume_ratio,
            pitch_ratio=request.pitch_ratio,
            text_type=request.text_type,
            with_frontend=request.with_frontend,
            frontend_type=request.frontend_type,
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=str(result["error"]))

        # Decode the base64 audio data
        audio_data = base64.b64decode(result["audio_data"])

        # Return the audio file
        return Response(
            content=audio_data,
            media_type=f"audio/{request.encoding}",
            headers={
                "Content-Disposition": (
                    f"attachment; filename=tts_output.{request.encoding}"
                )
            },
        )

    except Exception as e:
        logger.exception(f"Error in TTS endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/podcast/generate")
async def generate_podcast(request: GeneratePodcastRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_podcast_graph()
        final_state = workflow.invoke({"input": report_content})
        audio_bytes = final_state["output"]
        return Response(content=audio_bytes, media_type="audio/mp3")
    except Exception as e:
        logger.exception(f"Error occurred during podcast generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/ppt/generate")
async def generate_ppt(request: GeneratePPTRequest):
    try:
        report_content = request.content
        print(report_content)
        workflow = build_ppt_graph()
        final_state = workflow.invoke({"input": report_content})
        generated_file_path = final_state["generated_file_path"]
        with open(generated_file_path, "rb") as f:
            ppt_bytes = f.read()
        return Response(
            content=ppt_bytes,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )
    except Exception as e:
        logger.exception(f"Error occurred during ppt generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/prose/generate")
async def generate_prose(request: GenerateProseRequest):
    try:
        sanitized_prompt = request.prompt.replace("\r\n", "").replace("\n", "")
        logger.info(f"Generating prose for prompt: {sanitized_prompt}")
        workflow = build_prose_graph()
        events = workflow.astream(
            {
                "content": request.prompt,
                "option": request.option,
                "command": request.command,
            },
            stream_mode="messages",
            subgraphs=True,
        )
        return StreamingResponse(
            (f"data: {event[0].content}\n\n" async for _, event in events),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception(f"Error occurred during prose generation: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/prompt/enhance")
async def enhance_prompt(request: EnhancePromptRequest):
    try:
        sanitized_prompt = request.prompt.replace("\r\n", "").replace("\n", "")
        logger.info(f"Enhancing prompt: {sanitized_prompt}")

        # Convert string report_style to ReportStyle enum
        report_style = None
        if request.report_style:
            try:
                # Handle both uppercase and lowercase input
                style_mapping = {
                    "ACADEMIC": ReportStyle.ACADEMIC,
                    "POPULAR_SCIENCE": ReportStyle.POPULAR_SCIENCE,
                    "NEWS": ReportStyle.NEWS,
                    "SOCIAL_MEDIA": ReportStyle.SOCIAL_MEDIA,
                }
                report_style = style_mapping.get(
                    request.report_style.upper(), ReportStyle.ACADEMIC
                )
            except Exception:
                # If invalid style, default to ACADEMIC
                report_style = ReportStyle.ACADEMIC
        else:
            report_style = ReportStyle.ACADEMIC

        workflow = build_prompt_enhancer_graph()
        final_state = workflow.invoke(
            {
                "prompt": request.prompt,
                "context": request.context,
                "report_style": report_style,
            }
        )
        return {"result": final_state["output"]}
    except Exception as e:
        logger.exception(f"Error occurred during prompt enhancement: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.post("/api/mcp/server/metadata", response_model=MCPServerMetadataResponse)
async def mcp_server_metadata(request: MCPServerMetadataRequest):
    """Get information about an MCP server."""
    # Check if MCP server configuration is enabled
    if os.getenv("ENABLE_MCP_SERVER_CONFIGURATION", "false").lower() not in [
        "true",
        "1",
        "yes",
    ]:
        raise HTTPException(
            status_code=403,
            detail="MCP server configuration is disabled. Set ENABLE_MCP_SERVER_CONFIGURATION=true to enable MCP features.",
        )

    try:
        # Set default timeout with a longer value for this endpoint
        timeout = 300  # Default to 300 seconds for this endpoint

        # Use custom timeout from request if provided
        if request.timeout_seconds is not None:
            timeout = request.timeout_seconds

        # Load tools from the MCP server using the utility function
        tools = await load_mcp_tools(
            server_type=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            timeout_seconds=timeout,
        )

        # Create the response with tools
        response = MCPServerMetadataResponse(
            transport=request.transport,
            command=request.command,
            args=request.args,
            url=request.url,
            env=request.env,
            tools=tools,
        )

        return response
    except Exception as e:
        logger.exception(f"Error in MCP server metadata endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=INTERNAL_SERVER_ERROR_DETAIL)


@app.get("/api/rag/config", response_model=RAGConfigResponse)
async def rag_config():
    """Get the config of the RAG."""
    return RAGConfigResponse(provider=SELECTED_RAG_PROVIDER)


@app.get("/api/rag/resources", response_model=RAGResourcesResponse)
async def rag_resources(request: Annotated[RAGResourceRequest, Query()]):
    """Get the resources of the RAG."""
    retriever = build_retriever()
    if retriever:
        return RAGResourcesResponse(resources=retriever.list_resources(request.query))
    return RAGResourcesResponse(resources=[])


@app.get("/api/config", response_model=ConfigResponse)
async def config():
    """Get the config of the server."""
    return ConfigResponse(
        rag=RAGConfigResponse(provider=SELECTED_RAG_PROVIDER),
        models=get_configured_llm_models(),
    )
