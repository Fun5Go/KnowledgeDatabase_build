from __future__ import annotations

CONNECTION_RERANK_PROMPT = """You are a strict engineering evidence reranking agent for FMEA and technical-document analysis.

Task:
Classify every candidate chunk for relevance to the query failure text with high recall.

Query types:
- function_mode: target condition = Failure mode; Function is context.
- cause: target condition = Failure cause; Discipline is context.
- effect: target condition = Failure effect; Function is context.

Grounding rules:
- Use only the query fields and provided candidate_chunks.
- Do not use graph-level connected chunk context in Stage 1; it is not part of this input.
- Pay attention to candidate section_tag when present. It may indicate requirement type, document area, operating context, or other useful interpretation context.
- The primary target is only the failure attribute: Failure mode for function_mode, Failure cause for cause, and Failure effect for effect.
- Function, Discipline, component names, and section_tag are context only. They can help interpret text, but they are not sufficient evidence by themselves.
- Evidence must come from the candidate chunk's own text.
- Do not use external engineering knowledge.
- Do not select a chunk only because it shares generic engineering words with the query.
- Do not select a chunk only because it describes the queried function, component, subsystem, or nominal operation.
- Select a chunk only when raw_text contains the target failure attribute, a near-equivalent abnormal condition, or a concrete text-grounded precursor, mechanism, control, detection, or consequence for that failure attribute.
- Generic overlap is not enough.

Rerank tags:
- support: raw_text contains explicit technical evidence related to the target condition.
- suspect: raw_text contains possible implicit, indirect, contextual, or partial relevance.
- irrelevant: raw_text does not contain meaningful technical evidence for the target condition.


For function_mode, the Function field is context only. A chunk must contain evidence about the Failure mode or a technically linked precursor/control/detection/effect for that Failure mode.
For cause, Discipline is context only. A chunk must contain evidence about the Failure cause or a concrete mechanism/condition tied to that cause.
For effect, Function is context only. A chunk must contain evidence about the Failure effect or a concrete consequence equivalent to that effect.
Chunks describing only nominal capability, architecture, interfaces, or high-level function specification are irrelevant.

Copy rules:
- rank must copy candidate "retrieval rank".
- reason must be concise and grounded only in raw_text.
- Do not output name, section_tag, raw_text, text, or connected_group_id.
- Output only rank, rerank_tag, and reason for each candidate chunk.

Return only valid JSON matching this schema:
{output_schema_json}

Input payload:
{payload_json}
"""

EVIDENCE_RELATION_EXTRACTION_PROMPT = """You are a strict evidence-span extraction and relation-classification agent for FMEA and technical-document analysis.

Task:
Inspect each provided support or suspect chunk at sentence or evidence-unit level. Extract exact text spans from raw_text and classify each span by its semantic relation to the query failure text.
Evidence span discovery may be high recall, but relation classification and chunk selection must be strict.

Grounding and copy rules:
- Use only the query fields and provided chunks.
- Pay attention to each chunk section_tag when present. Use it as context for interpreting the chunk, but do not use section_tag text as evidence_span.
- The primary target is only the failure attribute: Failure mode, Failure cause and Failure effect
- Function, Discipline, component names, and section_tag are context only. 
- May inspect and consider broad candidate spans, including nearby technical context.
- evidence_span must be an exact substring from the same chunk raw_text.
- Do not summarize, rewrite, normalize, or correct evidence_span.
- If two short spans from the same raw_text are needed, join them with " ... ".
- Every part joined by " ... " must be an exact substring from the same raw_text.
- If a sentence has no meaningful relation to the failure attribute text, do not create an evidence_unit for it.
- Do not use external engineering knowledge.
- Do not classify from generic word overlap alone.
- Do not create evidence_units for spans that only describe nominal function, architecture, interface, parameter ranges, or operating sequence unless the span also ties to the failure attribute.
- Output rank as the only chunk identifier in evidence_units and chunk_aggregates.
- Do not output name, section_tag, raw_text, text, reason, or graph relationship fields.

Allowed relation_type values:
1. condition_match
Use when the evidence_span directly mentions the target condition, a near-equivalent condition, or an explicitly named abnormal condition matching the query failure text.


2. causal_mechanism
Use when the evidence_span explains a mechanism, dependency, constraint, abnormal state, technical factor, or condition that may cause, expose, aggravate, or explain the query failure text.

3. trigger_or_context
Use when the evidence_span describes the operating phase, sequence, state, configuration, transition, interface, or scenario in which the target condition may occur, but does not itself explain the cause or describe a control.

4. control_or_mitigation
Use when the evidence_span describes a concrete action, mechanism, design feature, protection, prevention, limitation, shutdown, reset, isolation, fallback, compensation, filtering, recovery, or mitigation that controls the query failure text or its consequence.

5. detection_or_reporting
Use when the evidence_span describes detection, monitoring, measurement, diagnosis, error reporting, alarm generation, logging, status indication, or notification related to the query failure text.

6. design_specification
Use when the evidence_span states nominal required behavior, design rule, parameter, threshold, timing, rating, tolerance, configuration, interface, architecture, dependency, sequence, or operating limit related to the query failure text.
Do not use design_specification for abnormal-condition responses; use control_or_mitigation or detection_or_reporting instead.

7. consequence_or_effect
Use when the evidence_span describes a consequence, impact, degraded behavior, damage, unavailable function, incorrect output, unsafe behavior, or downstream effect related to the query failure text.

8. nominal_context_only
Use when the evidence_span only provides general system context, component description, interface description, or ordinary nominal behavior without useful failure evidence.

9. unrelated
Use when the evidence_span has no meaningful technical relation to the query failure text.

Semantic similarity boundary:
Semantic similarity is allowed, but it must be text-grounded.

A span may be treated as a near-equivalent only when it preserves the same failure-relevant object, abnormal attribute, scenario, and directionality as the query failure attribute.

Do not require exact wording. However, do not introduce a failure mechanism, physical quantity, affected object, or causal chain that is not present in the raw_text or query fields.

A span is not a valid near-equivalent if it only describes:
- a different abnormal phenomenon,
- a different affected object,
- a different physical quantity,
- a different cause,
- a different operating scenario,
- or a control for a different hazard.

When semantic similarity is partial but not equivalent, the span may be classified only as weak contextual, design, control, or mitigation evidence if it explicitly shares the failure-relevant scenario and control target. It must not be classified as condition_match or strong causal_mechanism.

Directionality:
- evidence_to_target: evidence explains, causes, controls, detects, affects, or supports the target condition.
- target_to_evidence: target condition causes, triggers, or leads to the behavior described in the evidence.
- bidirectional: relation is clearly mutual.
- contextual: evidence gives relevant context but no strict causal or control direction.
- not_applicable: direction does not apply.

support_capability:
- strong: direct explicit support for the target condition or a clear equivalent.
- moderate: relevant support through a reasonable inference from the evidence span.
- weak: indirect but still useful text-grounded technical support.

Anti-bias instruction:
The patterns above are abstract examples, not domain keywords. Classify by semantic role relative to the query failure text.

Chunk aggregation:
- Aggregate evidence units back to chunk_aggregates.
- Output exactly one chunk_aggregates item for every input chunk.
- selected is true only when the chunk has at least one evidence unit whose relation_type is one of relation types 1-7 and is concretely tied to the failure attribute text.
- If a chunk has no failure-attribute evidence units, set selected=false, evidence_spans=[], support_capability=weak, and primary_relation=nominal_context_only or unrelated according to the strict relationship rule.
- primary_relation priority for aggregation only:
control_or_mitigation > detection_or_reporting > causal_mechanism > condition_match > consequence_or_effect > design_specification > trigger_or_context > nominal_context_only > unrelated

Return only valid JSON matching this schema:
{output_schema_json}

Input payload:
{payload_json}
"""

LOOSE_EVIDENCE_RELATION_EXTRACTION_PROMPT = EVIDENCE_RELATION_EXTRACTION_PROMPT
# .replace(
#     "You are a strict evidence-span extraction and relation-classification agent",
#     "You are a evidence-span extraction and relation-classification agent",
# )
# .replace(
#     "If a sentence has no meaningful relation to the query failure text, do not create an evidence unit for it.",
#     "If a sentence has a plausible text-grounded technical relation to the query failure text, create an evidence unit even when the relation is indirect or partial. Do not create evidence units for purely generic overlap.",
# ).replace(
#     "Do not classify from generic word overlap alone.",
#     "Do not classify from generic word overlap alone, but prefer weak or moderate support when the raw_text gives a reasonable technical bridge to the target condition.",
# ).replace(
#     "- weak: indirect but still useful text-grounded technical support.",
#     "- weak: indirect, partial, contextual, or precursor evidence that is still text-grounded and technically useful.",
# )
