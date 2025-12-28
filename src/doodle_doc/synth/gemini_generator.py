from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from google.genai import Client

SUBJECTS = [
    "linear algebra lecture notes with matrices and vectors",
    "organic chemistry reaction mechanisms with benzene rings",
    "electrical circuit diagrams with resistors and capacitors",
    "data structures showing binary trees and linked lists",
    "calculus derivatives and integrals with graphs",
    "physics force diagrams and free body diagrams",
    "economics supply and demand curves",
    "biology cell diagrams with organelles",
    "geometry proofs with triangles and circles",
    "statistics notes with probability distributions",
    "computer architecture diagrams with CPU and memory",
    "thermodynamics equations with heat flow diagrams",
]


@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-flash-image"
    seed: int = 42


class GeminiGenerator:
    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig()
        self._client: Client | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def generate_notes_page(self, subject: str) -> Image.Image:
        prompt = f"""Generate an image of a page of handwritten student notes about {subject}.
Include diagrams, equations, annotations, and text scattered across the page.
Black ink on white paper, realistic handwriting style.
Make it look like authentic student notes with some messiness and corrections."""
        return self._generate(prompt)

    def generate_doodle_for_page(self, page_image: Image.Image) -> tuple[Image.Image, str]:
        prompt = """Look at this page of notes. Pick one specific visual element - a symbol, diagram, shape, or equation - that would be distinctive and searchable.

Generate a quick hand-drawn doodle of ONLY that element. Make it:
- Rough pencil sketch style
- Imperfect wobbly lines
- Like someone drawing quickly from memory
- Just the one element, nothing else

After the image, write a short description of what element you picked."""

        img_bytes = io.BytesIO()
        page_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        from google.genai import types

        response = self.client.models.generate_content(
            model=self.config.model,
            contents=[
                types.Part.from_bytes(data=img_bytes.read(), mime_type="image/png"),
                prompt,
            ],
        )

        doodle_image = None
        description = ""

        for part in response.parts:
            if part.inline_data:
                doodle_image = self._to_pil(part.as_image())
            elif part.text:
                description = part.text.strip()

        if doodle_image is None:
            raise RuntimeError("No image in Gemini response")

        return doodle_image, description

    def _generate(self, prompt: str) -> Image.Image:
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
        )

        for part in response.parts:
            if part.inline_data:
                return self._to_pil(part.as_image())

        raise RuntimeError("No image in Gemini response")

    def _to_pil(self, img: Any) -> Image.Image:
        return Image.open(io.BytesIO(img.image_bytes)).convert("RGB")
