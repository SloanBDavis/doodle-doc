from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from google.genai import Client


@dataclass(frozen=True)
class PageArchetype:
    key: str
    domain: str
    subject: str
    layout: str
    paper_style: str
    writing_style: str


PAGE_ARCHETYPES = [
    PageArchetype(
        key="calculus_problem_set",
        domain="math",
        subject="calculus derivatives, integrals, tangent lines, and curve sketches",
        layout="a worked problem set with numbered exercises, partial solutions, boxed answers, and margin notes",
        paper_style="faint dot-grid tablet paper with a light toolbar crop like a note-taking app export",
        writing_style="messy but legible black handwriting with a few blue highlights and red corrections",
    ),
    PageArchetype(
        key="physics_worksheet",
        domain="science",
        subject="physics force diagrams, kinematics equations, and free-body sketches",
        layout="a printed worksheet with typed questions and handwritten annotations layered on top",
        paper_style="white worksheet paper with subtle scan shadows and occasional underlines",
        writing_style="quick stylus handwriting, arrows, circled results, and crossed-out mistakes",
    ),
    PageArchetype(
        key="chemistry_notes",
        domain="science",
        subject="organic chemistry mechanisms, benzene rings, reaction arrows, and reagent labels",
        layout="lecture notes with short text blocks, reaction sequences, and side comments",
        paper_style="clean white notebook paper with light highlighter streaks",
        writing_style="compact handwriting with slightly uneven spacing and hand-drawn molecular diagrams",
    ),
    PageArchetype(
        key="biology_handout",
        domain="science",
        subject="cell biology diagrams, labeled organelles, microscopy callouts, and process arrows",
        layout="an annotated handout with printed figure captions and handwriting on top",
        paper_style="tablet-export page with faint off-white background and pasted image areas",
        writing_style="mixed neat labels and hurried note fragments with colored emphasis marks",
    ),
    PageArchetype(
        key="linear_algebra_notes",
        domain="math",
        subject="linear algebra matrices, determinants, eigenvectors, and vector geometry",
        layout="dense lecture notes with formulas, matrix blocks, and a few small plots",
        paper_style="graph paper with faint gray lines",
        writing_style="dark stylus handwriting with some smudgy erasures and rewritten terms",
    ),
    PageArchetype(
        key="statistics_review",
        domain="math",
        subject="statistics distributions, confidence intervals, hypothesis tests, and scatter plots",
        layout="review sheet with headings, mini charts, and worked examples",
        paper_style="light cream ruled paper with slight tablet-export cropping",
        writing_style="clean headings plus rough handwritten derivations and arrows between ideas",
    ),
]


@dataclass
class GeminiConfig:
    model: str = "gemini-3.1-flash-image-preview"
    prompt_version: str = "v2"


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

    def build_page_prompt(self, archetype: PageArchetype) -> str:
        return f"""Create a single realistic page of student notes that feels like a Notability or tablet note export.

Subject matter: {archetype.subject}.
Page archetype: {archetype.layout}.
Paper style: {archetype.paper_style}.
Handwriting style: {archetype.writing_style}.

Requirements:
- Portrait page with lots of authentic handwritten math or science content
- Include diagrams, symbols, arrows, labels, small mistakes, corrections, and uneven spacing
- Mix clean structure with natural messiness; avoid perfect textbook layout
- Some archetypes should feel like handwriting layered over printed worksheet or handout content
- Keep the page information-dense and retrieval-friendly with distinct visual elements
- White or lightly tinted background, no decorative borders, no extra surrounding desk or objects
- Make it look like something a student actually studied from
"""

    def build_doodle_prompt(self) -> str:
        return """Look at this notes page and choose one visually distinctive element that would be useful for sketch-based retrieval.

Generate a doodle of ONLY that element on a plain white background.

Requirements for the doodle:
- rough quick hand-drawn sketch
- black or graphite lines only
- no surrounding page context
- no extra labels unless the selected element inherently includes text
- centered with generous whitespace
- simple enough that it looks like a retrieval query, not a full diagram redraw

After the image, write one short line describing the chosen element."""

    def generate_notes_page(self, archetype: PageArchetype) -> Image.Image:
        return self._generate(self.build_page_prompt(archetype))

    def generate_doodle_for_page(self, page_image: Image.Image) -> tuple[Image.Image, str]:
        img_bytes = io.BytesIO()
        page_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        from google.genai import types

        response = self.client.models.generate_content(
            model=self.config.model,
            contents=[
                types.Part.from_bytes(data=img_bytes.read(), mime_type="image/png"),
                self.build_doodle_prompt(),
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
            raise RuntimeError("No image in model response")

        return doodle_image, description

    def _generate(self, prompt: str) -> Image.Image:
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
        )

        for part in response.parts:
            if part.inline_data:
                return self._to_pil(part.as_image())

        raise RuntimeError("No image in model response")

    def _to_pil(self, img: Any) -> Image.Image:
        return Image.open(io.BytesIO(img.image_bytes)).convert("RGB")
