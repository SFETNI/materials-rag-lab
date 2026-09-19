# Runtime Query Decomposer v1

Decide whether the question contains multiple independently retrievable factual/evidence requirements.

Return exactly one JSON object:
{
  "question_id": "...",
  "original_question": "...",
  "decomposable": true,
  "subqueries": ["...", "..."]
}

Rules:
- If decomposable=false, subqueries must be [].
- If decomposable=true, produce 2 to 4 unique subqueries.
- Each subquery must target a distinct evidence requirement.
- Each subquery must be understandable independently.
- Preserve specimen IDs, build IDs, document IDs, dates, quantities, stress definitions and materials terminology.
- Do not answer the question.
- Do not invent facts.
- Do not name an expected source/document unless already named in the question.
- Do not merely paraphrase the whole question several ways.
- No tool use, files, web search, repository access or outside information.
- Return JSON only.
