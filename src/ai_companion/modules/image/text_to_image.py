import base64
import logging
import os
from typing import Optional
from io import BytesIO

from ai_companion.core.exceptions import TextToImageError
from ai_companion.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from ai_companion.settings import settings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# ✅ Correct imports from Google docs
from google import genai
from google.genai import types
from PIL import Image

class ScenarioPrompt(BaseModel):
    """Class for the scenario response"""
    narrative: str = Field(..., description="The AI's narrative response to the question")
    image_prompt: str = Field(..., description="The visual prompt to generate an image representing the scene")

class EnhancedPrompt(BaseModel):
    """Class for the text prompt"""
    content: str = Field(..., description="The enhanced text prompt to generate an image")

class TextToImage:
    """A class to handle text-to-image generation using Google's Gemini image model."""

    REQUIRED_ENV_VARS = ["GROQ_API_KEY", "GEMINI_API_KEY"]

    def __init__(self):
        """Initialize the TextToImage class and validate environment variables."""
        self._validate_env_vars()
        self._client: Optional[genai.Client] = None
        self.logger = logging.getLogger(__name__)

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def client(self) -> genai.Client:
        """Get or create Google Genai client instance using singleton pattern."""
        if self._client is None:
            # ✅ Correct client initialization from Google docs
            self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        return self._client

    async def generate_image(self, prompt: str, output_path: str = "") -> bytes:
        """Generate an image using Google's Gemini image model."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        try:
            self.logger.info(f"Generating image for prompt: '{prompt}'")

            # ✅ Use exact structure from Google docs
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image-preview",  # ✅ Correct model name
                contents=[prompt],  # ✅ Correct format
            )

            # ✅ Parse response exactly like Google docs
            image_data = None
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    self.logger.info(f"Text response: {part.text}")
                elif part.inline_data is not None:
                    # ✅ Extract image data exactly like docs
                    image_data = part.inline_data.data
                    break

            if image_data is None:
                raise TextToImageError("No image data found in response")

            # Save image if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # ✅ Use PIL to save like in docs
                image = Image.open(BytesIO(image_data))
                image.save(output_path)
                self.logger.info(f"Image saved to {output_path}")

            return image_data

        except Exception as e:
            raise TextToImageError(f"Failed to generate image: {str(e)}") from e

    # Keep your existing methods unchanged
    async def create_scenario(self, chat_history: list = None) -> ScenarioPrompt:
        """Creates a first-person narrative scenario and corresponding image prompt based on chat history."""
        try:
            formatted_history = "\n".join([f"{msg.type.title()}: {msg.content}" for msg in chat_history[-5:]])

            self.logger.info("Creating scenario from chat history")

            llm = ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.4,
                max_retries=2,
            )

            structured_llm = llm.with_structured_output(ScenarioPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["chat_history"],
                    template=IMAGE_SCENARIO_PROMPT,
                )
                | structured_llm
            )

            scenario = chain.invoke({"chat_history": formatted_history})
            self.logger.info(f"Created scenario: {scenario}")

            return scenario

        except Exception as e:
            raise TextToImageError(f"Failed to create scenario: {str(e)}") from e

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance a simple prompt with additional details and context."""
        try:
            self.logger.info(f"Enhancing prompt: '{prompt}'")

            llm = ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.25,
                max_retries=2,
            )

            structured_llm = llm.with_structured_output(EnhancedPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["prompt"],
                    template=IMAGE_ENHANCEMENT_PROMPT,
                )
                | structured_llm
            )

            enhanced_prompt = chain.invoke({"prompt": prompt}).content
            self.logger.info(f"Enhanced prompt: '{enhanced_prompt}'")

            return enhanced_prompt

        except Exception as e:
            raise TextToImageError(f"Failed to enhance prompt: {str(e)}") from e
