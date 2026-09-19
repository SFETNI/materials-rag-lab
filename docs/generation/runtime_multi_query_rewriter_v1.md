# Runtime Multi-query Rewriter v1

Generate exactly three alternative search queries for the supplied question.

Return exactly one JSON object:
{
  "question_id": "...",
  "original_query": "...",
  "rewrites": ["...", "...", "..."]
}

Rules:
- Preserve technical identifiers, quantities and dates.
- Preserve factual intent.
- Vary wording enough to expose different retrieval matches.
- Do not answer the question.
- Do not invent facts.
- Do not cite sources.
- Do not decompose into separately answered subquestions.
- Exactly three unique non-empty rewrites.
- No tool use, files, web search, repository access or outside information.
- Return JSON only.
