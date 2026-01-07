import sys
import asyncio
import base64
import re
from pathlib import Path
import aiohttp
import os 
import json
from PIL import Image
import io
import time
import numpy as np

llm_kwargs = {
    "model": "zai-org/GLM-4.1V-9B-Thinking",
    "base_url": "http://localhost:8001/v1/chat/completions",  # Changed to local VLLM
    "api_key": "EMPTY",  # VLLM typically uses "EMPTY" or no auth
    "temperature": 0.0,
    "max_tokens": 1024,
    "sampling_params": {"seed": 42},
}

SYSTEM_PROMPT = "You are a specialized 3D model evaluation system. Analyze visual quality and prompt adherence with expert precision. Always respond with valid JSON only."
USER_PROMPT_IMAGE = """Does each 3D model match the image prompt?

Penalty 0-10:
0 = Perfect match
3 = Minor issues (slight shape differences, missing small details)
5 = Moderate issues (wrong style, significant details missing)
7 = Major issues (wrong category but related, e.g. chair vs stool)
10 = Completely wrong object

Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}"""


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response content, handling markdown code blocks."""
    # Handle case where response might have markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from the content
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            raise Exception(f"Failed to parse JSON from response: {content}")
    
    return {
        "penalty_1": parsed.get("penalty_1", 10),
        "penalty_2": parsed.get("penalty_2", 10),
        "issues": parsed.get("issues", "N/A"),
    }


async def judge_duel_async(
    prompt_b64: str,
    render1: str,
    render2: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    sampling_params: dict | None = None,
) -> dict:
    """
    Async version: Call local VLLM API to judge a duel between two 3D model renders.
    Returns dict with penalty_1, penalty_2, and issues.
    """
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image prompt to generate 3D model:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{prompt_b64}"},
                },
                {"type": "text", "text": "First 3D model (4 different views):"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{render1}"},
                },
                {"type": "text", "text": "Second 3D model (4 different views):"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{render2}"},
                },
                {"type": "text", "text": USER_PROMPT_IMAGE},
            ],
        },
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=base_url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"VLLM API error: {response.status} - {text}")
            
            result = await response.json()
    
    content = result["choices"][0]["message"]["content"]
    return _parse_json_response(content)


def judge_duel(
    prompt_b64: str,
    render1: str,
    render2: str,
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    sampling_params: dict | None = None,
) -> dict:
    """
    Sync version: Call local VLLM API to judge a duel between two 3D model renders.
    Returns dict with penalty_1, penalty_2, and issues.
    
    Note: This is kept for backward compatibility. Use judge_duel_async for better performance.
    """
    import requests
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image prompt to generate 3D model:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{prompt_b64}"},
                },
                {"type": "text", "text": "First 3D model (4 different views):"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{render1}"},
                },
                {"type": "text", "text": "Second 3D model (4 different views):"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{render2}"},
                },
                {"type": "text", "text": USER_PROMPT_IMAGE},
            ],
        },
    ]
    
    response = requests.post(
        url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }),
        timeout=10,
    )
    
    if response.status_code != 200:
        raise Exception(f"VLLM API error: {response.status_code} - {response.text}")
    
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    return _parse_json_response(content)



def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


async def judge_3d_duel(render_left: Image.Image, render_right: Image.Image, prompt: Image.Image):
    """
    Run a position-balanced duel evaluation between two 3D model renders.
    Performs both direct and swapped comparisons in parallel for fairness (avoid position bias).
    
    Args:
        render_left: PIL Image of the first render
        render_right: PIL Image of the second render  
        prompt: PIL Image of the original prompt
        
    Returns:
        tuple: (winner, avg_penalty_left, avg_penalty_right, combined_issues)
    """
    # Convert all images to base64
    render_left_b64 = _image_to_base64(render_left)
    render_right_b64 = _image_to_base64(render_right)
    prompt_b64 = _image_to_base64(prompt)

    try:
        s0 = time.time()
        # Run both passes in parallel (position-balanced duel)
        result_1, result_2 = await asyncio.gather(
            judge_duel_async(
                render1=render_left_b64, 
                render2=render_right_b64, 
                prompt_b64=prompt_b64, 
                **llm_kwargs
            ),
            judge_duel_async(
                render1=render_right_b64, 
                render2=render_left_b64, 
                prompt_b64=prompt_b64, 
                **llm_kwargs
            ),
        )
        print(f"Time process {time.time() - s0}")
        # Calculate averaged penalties (swap positions in result_2)
        left_penalty = (result_1["penalty_1"] + result_2["penalty_2"]) / 2
        right_penalty = (result_1["penalty_2"] + result_2["penalty_1"]) / 2
        
        # Determine winner based on penalty difference
        if abs(left_penalty - right_penalty) <= 1:
            winner = "draw"
        elif left_penalty < right_penalty:
            winner = "left"
        else:
            winner = "right"
        
        combined_issues = f"Pass 1: {result_1.get('issues', 'N/A')}. Pass 2: {result_2.get('issues', 'N/A')}"
        
        return winner, left_penalty, right_penalty, combined_issues
        
    except Exception as e:
        # Return draw on error (matching judge-service behavior)
        return "draw", 10, 10, f"Internal error: {str(e)}"

async def judge_question(
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
) -> dict:
    """
    Async version: Call local VLLM API to judge a duel between two 3D model renders.
    Returns dict with penalty_1, penalty_2, and issues.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image prompt:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    
    payload = {
        "model": llm_kwargs["model"],
        "messages": messages,
        "temperature": llm_kwargs["temperature"],
        "max_tokens": llm_kwargs["max_tokens"],
    }
    
    headers = {
        "Authorization": f"Bearer {llm_kwargs['api_key']}",
        "Content-Type": "application/json",
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url=llm_kwargs["base_url"],
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"VLLM API error: {response.status} - {text}")
            
            result = await response.json()
    
    content = result["choices"][0]["message"]["content"]
    return _parse_json_response(content)

def calculate_psnr(image1: Image.Image, image2: Image.Image) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        
    Returns:
        float: PSNR value in dB (higher is better, typical range 20-50)
    """
    
    # Convert images to same size if needed
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    
    # Convert to RGB if needed
    if image1.mode != 'RGB':
        image1 = image1.convert('RGB')
    if image2.mode != 'RGB':
        image2 = image2.convert('RGB')
    
    # Convert to numpy arrays
    img1_array = np.array(image1, dtype=np.float64)
    img2_array = np.array(image2, dtype=np.float64)
    
    # Calculate MSE
    mse = np.mean((img1_array - img2_array) ** 2)
    
    if mse == 0:
        return float('inf')  # Images are identical
    
    # Calculate PSNR (assuming 8-bit images with max value 255)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr


def calculate_ssim(image1: Image.Image, image2: Image.Image) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        image1: First PIL Image
        image2: Second PIL Image
        
    Returns:
        float: SSIM value between -1 and 1 (1 means identical images)
    """
    
    # Convert images to same size if needed
    if image1.size != image2.size:
        image2 = image2.resize(image1.size)
    
    # Convert to grayscale for SSIM calculation
    if image1.mode != 'L':
        image1 = image1.convert('L')
    if image2.mode != 'L':
        image2 = image2.convert('L')
    
    # Convert to numpy arrays
    img1 = np.array(image1, dtype=np.float64)
    img2 = np.array(image2, dtype=np.float64)
    
    # SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    # Calculate means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Calculate variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]
    
    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / denominator
    
    return ssim

async def main():
    prompt_path = "assets/house.jpg"
    prompt = Image.open(prompt_path)

    response = await judge_question(user_prompt="Is this a house?", system_prompt="You are a specialized 3D model evaluation system. Analyze visual quality and prompt adherence with expert precision. Always respond with valid JSON only.", image_base64=_image_to_base64(prompt))
    
    print(f"Response: {response}")


if __name__ == "__main__":
    
    asyncio.run(main())
    
