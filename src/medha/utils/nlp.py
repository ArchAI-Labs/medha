"""NLP utilities: parameter extraction and keyword overlap scoring."""

import logging
import re
from collections import defaultdict

from medha.exceptions import ParameterExtractionError
from medha.types import QueryTemplate

logger = logging.getLogger(__name__)

_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at",
    "to", "for", "of", "with", "by",
})

# Words that look capitalized in a question but are never parameter values
_HEURISTIC_EXCLUDED_WORDS = frozenset({
    "show", "find", "get", "count", "list", "display", "fetch", "return",
    "give", "tell", "describe", "select", "what", "how", "which", "where",
    "when", "who", "are", "is", "do", "does", "have", "has", "can",
    "all", "top", "avg", "average", "total", "sum", "max", "min",
})


class ParameterExtractor:
    """Extract parameters from user questions using a cascading strategy."""

    _ENTITY_CACHE_MAXSIZE = 256
    _GLINER_DEFAULT_MODEL = "urchade/gliner_medium-v2.1"

    def __init__(
        self,
        use_spacy: bool = True,
        use_gliner: bool = False,
        gliner_model: str = _GLINER_DEFAULT_MODEL,
    ):
        """Initialize the extractor.

        Args:
            use_spacy: If True, attempt to load a spaCy model for NER.
                Falls back gracefully if spaCy is not installed.
            use_gliner: If True, attempt to load a GLiNER model for zero-shot NER.
                GLiNER uses template parameter names directly as entity labels,
                removing the need for hardcoded label mappings.
                Falls back gracefully if gliner is not installed.
            gliner_model: HuggingFace model ID for GLiNER. Defaults to
                ``urchade/gliner_medium-v2.1``.
        """
        self._nlp = None
        self._spacy_available = False
        self._gliner = None
        self._gliner_available = False
        # Cache entity extraction results: same question is parsed once
        # even when matched against many templates in the same search call.
        self._entity_cache: dict[str, dict[str, list[str]]] = {}
        # Cache GLiNER predictions: same (question, labels) tuple is inferred once,
        # making repeated calls (e.g. benchmarks) equivalent to spaCy's _entity_cache.
        self._gliner_cache: dict[tuple[str, tuple[str, ...]], dict[str, str]] = {}

        if use_spacy:
            self._try_load_spacy()
        if use_gliner:
            self._try_load_gliner(gliner_model)

    @property
    def spacy_available(self) -> bool:
        """Whether spaCy NER is available."""
        return self._spacy_available

    @property
    def gliner_available(self) -> bool:
        """Whether GLiNER zero-shot NER is available."""
        return self._gliner_available

    def extract(self, question: str, template: QueryTemplate) -> dict[str, str]:
        """Extract all parameters for a template from a question.

        Cascading strategy:
            1. Regex patterns from template.parameter_patterns
            2. GLiNER zero-shot NER (if enabled) — uses param names as labels
            3. spaCy NER (if enabled)
            4. Fallback heuristics (numbers, capitalized words)

        Args:
            question: The raw user question.
            template: The matched QueryTemplate.

        Returns:
            Dict mapping parameter names to extracted values.
            Only fully extracted results are returned (all params present).

        Raises:
            ParameterExtractionError: If not all required parameters
                can be extracted.
        """
        if not template.parameters:
            logger.debug("No parameters required for template '%s'", template.intent)
            return {}

        logger.debug(
            "Extracting params for template '%s': required=%s",
            template.intent,
            template.parameters,
        )

        # 1. Regex patterns from template
        params = self._extract_via_regex(question, template)
        if params:
            logger.debug("Regex extracted: %s", params)
        if len(params) == len(template.parameters):
            logger.debug("All params resolved via regex")
            return params

        # 2. GLiNER zero-shot NER (uses template param names as labels directly)
        if self._gliner_available:
            gliner_params = self._extract_via_gliner(question, template)
            if gliner_params:
                logger.debug("GLiNER extracted: %s", gliner_params)
            for key, value in gliner_params.items():
                if key not in params:
                    params[key] = value
            if len(params) == len(template.parameters):
                logger.debug("All params resolved via regex+GLiNER")
                return params

        # 3. spaCy NER
        if self._spacy_available:
            spacy_params = self._extract_via_spacy(question, template)
            if spacy_params:
                logger.debug("spaCy extracted: %s", spacy_params)
            for key, value in spacy_params.items():
                if key not in params:
                    params[key] = value
            if len(params) == len(template.parameters):
                logger.debug("All params resolved via regex+spaCy")
                return params

        # 4. Heuristic fallback
        heuristic_params = self._extract_via_heuristics(question, template)
        if heuristic_params:
            logger.debug("Heuristic extracted: %s", heuristic_params)
        for key, value in heuristic_params.items():
            if key not in params:
                params[key] = value

        if len(params) == len(template.parameters):
            logger.debug("All params resolved via heuristics fallback")
            return params

        missing = set(template.parameters) - set(params)
        logger.warning(
            "Incomplete extraction for template '%s': missing=%s, got=%s, cascade=%s",
            template.intent,
            missing,
            params,
            f"regex={'ok' if params else 'miss'} gliner={'ok' if self._gliner_available else 'off'} "
            f"spacy={'ok' if self._spacy_available else 'off'}",
        )
        raise ParameterExtractionError(
            f"Could not extract parameters {missing} from: {question!r}"
        )

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract named entities using spaCy + regex fallback.

        Results are cached per text: the same question is parsed only once
        even when matched against multiple templates in the same search call.

        Returns a dict mapping entity type to list of values:
            - "number": ["5", "100"]
            - "person": ["John Smith"]
            - "org": ["Acme Corp"]
            - "cardinal": ["5"]

        Always includes regex-extracted numbers and capitalized words,
        even when spaCy is available (union of both).
        """
        if text in self._entity_cache:
            logger.debug("Entity cache HIT for text len=%d", len(text))
            return self._entity_cache[text]

        entities: dict[str, list[str]] = defaultdict(list)

        if self._spacy_available and self._nlp is not None:
            doc = self._nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "CARDINAL"):
                    entities[ent.label_.lower()].append(ent.text)

        # Regex fallback (always applied — union with spaCy)
        numbers = re.findall(r"\b\d+\b", text)
        names = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b", text)

        if numbers:
            entities["number"].extend(numbers)
        if names:
            entities["person"].extend(names)

        result = dict(entities)

        # Evict all when full: simple and correct for a pure-function cache
        if len(self._entity_cache) >= self._ENTITY_CACHE_MAXSIZE:
            self._entity_cache.clear()
        self._entity_cache[text] = result

        return result

    def render_query(
        self, template: QueryTemplate, parameters: dict[str, str]
    ) -> str:
        """Inject parameters into a query template.

        Two paths, chosen per parameter:

        - If ``template.parameter_patterns`` declares a pattern for this
          parameter and the value ``re.fullmatch``-es it, the value has
          already been shape-checked by whoever wrote the template (a
          fullmatch is anchored to the whole value, not just a substring of
          the question) — it is injected as-is, without running it through
          the generic sanitizer. This is what lets values like ``10:00-12:00``
          or ``10/08/2026`` survive intact when a template opts into them.
        - Otherwise (no declared pattern, or the value doesn't match one —
          e.g. it came from NER or the heuristic fallback, which have no
          declared shape) the value goes through ``_sanitize_value()``. If
          sanitization would change the value, that value is never rendered:
          we raise instead of silently substituting the corrupted result.

        Args:
            template: The QueryTemplate with placeholders.
            parameters: Extracted parameter values.

        Returns:
            The rendered query string.

        Raises:
            ParameterExtractionError: If a non-pattern-verified value would be
                altered by sanitization (or is empty after it), or a
                placeholder remains unfilled.
        """
        query = template.query_template

        for param, value in parameters.items():
            if self._is_pattern_verified(template, param, value):
                safe_value = value
            else:
                safe_value = self._sanitize_value(value)
                if safe_value != value:
                    logger.warning(
                        "Parameter '%s' altered by sanitization (raw=%r, sanitized=%r); "
                        "refusing to render a corrupted query",
                        param, value, safe_value,
                    )
                    raise ParameterExtractionError(
                        f"Parameter {param!r} value {value!r} contains characters removed "
                        f"by sanitization (would render as {safe_value!r}); refusing to "
                        f"substitute a corrupted value. If {value!r} is a legitimate value, "
                        f"declare an anchored pattern for {param!r} in the template's "
                        f"parameter_patterns."
                    )
            if not safe_value:
                logger.warning(
                    "Parameter '%s' empty after sanitization (raw='%s')", param, value
                )
                raise ParameterExtractionError(
                    f"Parameter {param!r} value {value!r} is empty after sanitization"
                )
            query = query.replace(f"{{{param}}}", safe_value)

        # Check for remaining unfilled placeholders
        remaining = re.findall(r"\{(\w+)\}", query)
        if remaining:
            logger.warning("Unfilled placeholders after render: %s", remaining)
            raise ParameterExtractionError(
                f"Unfilled placeholders in rendered query: {remaining}"
            )

        logger.debug("Rendered query: '%s'", query[:80])
        return query

    # --- Private methods ---

    def _try_load_spacy(self) -> None:
        """Attempt to load a spaCy model. Try multiple models in order."""
        models = ["en_core_web_sm", "en_core_web_md", "en"]
        for model_name in models:
            try:
                import spacy

                self._nlp = spacy.load(model_name)
                self._spacy_available = True
                logger.info("Loaded spaCy model: %s", model_name)
                return
            except (OSError, ImportError):
                continue
        logger.info("spaCy not available; using regex-only extraction")

    def _try_load_gliner(self, model_name: str) -> None:
        """Attempt to load a GLiNER model from HuggingFace."""
        try:
            from gliner import GLiNER

            self._gliner = GLiNER.from_pretrained(model_name)
            self._gliner_available = True
            logger.info("Loaded GLiNER model: %s", model_name)
        except ImportError:
            logger.info("gliner package not installed; run: pip install gliner")
        except Exception as exc:
            logger.info("GLiNER model load failed (%s); skipping", exc)

    def _extract_via_gliner(
        self, question: str, template: QueryTemplate
    ) -> dict[str, str]:
        """Extract parameters using GLiNER zero-shot NER.

        Uses ``template.parameters`` directly as entity labels, so no hardcoded
        label mapping is needed. GLiNER returns the best span for each label.

        Results are cached per (question, labels) pair so repeated calls with
        the same inputs (e.g. benchmarks) avoid redundant Transformer inference.
        """
        if not template.parameters or self._gliner is None:
            return {}

        labels = list(template.parameters)
        cache_key = (question, tuple(sorted(labels)))
        if cache_key in self._gliner_cache:
            return self._gliner_cache[cache_key]

        try:
            entities = self._gliner.predict_entities(question, labels)
        except Exception as exc:
            logger.warning("GLiNER prediction failed: %s", exc)
            return {}

        params: dict[str, str] = {}
        for entity in entities:
            label: str = entity["label"]
            text: str = entity["text"]
            # GLiNER sometimes includes the label word as a prefix of the span
            # (e.g. label="project" → text="project Hermes"). Strip it.
            if text.lower().startswith(label.lower() + " "):
                text = text[len(label) + 1:]
            if label in template.parameters and label not in params:
                params[label] = text

        if len(self._gliner_cache) >= self._ENTITY_CACHE_MAXSIZE:
            self._gliner_cache.clear()
        self._gliner_cache[cache_key] = params
        return params

    def _extract_via_regex(
        self, question: str, template: QueryTemplate
    ) -> dict[str, str]:
        """Extract parameters using template-defined regex patterns."""
        params: dict[str, str] = {}
        if not template.parameter_patterns:
            return params

        for param_name, pattern in template.parameter_patterns.items():
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                match = matches[0]
                params[param_name] = match if isinstance(match, str) else match[0]

        return params

    def _extract_via_spacy(
        self, question: str, template: QueryTemplate
    ) -> dict[str, str]:
        """Extract parameters using spaCy NER."""
        entities = self.extract_entities(question)
        params: dict[str, str] = {}

        for param in template.parameters:
            if param in ("count", "number") and "number" in entities:
                params[param] = entities["number"][0]
            elif param in ("user", "person", "name") and "person" in entities:
                params[param] = entities["person"][0]
            elif param in ("company", "org", "organization") and "org" in entities:
                params[param] = entities["org"][0]
            elif param == "project":
                project_match = re.search(
                    r"project\s+([A-Za-z0-9_]+)", question, re.IGNORECASE
                )
                if project_match:
                    params[param] = project_match.group(1)

        return params

    def _extract_via_heuristics(
        self, question: str, template: QueryTemplate
    ) -> dict[str, str]:
        """Fallback: extract numbers and capitalized words."""
        params: dict[str, str] = {}
        numbers = re.findall(r"\b\d+\b", question)
        # Exclude common question/command words that are capitalized at sentence start
        # (e.g. "Show", "Find", "Count") — they are never meaningful parameter values.
        capitalized = [
            w for w in re.findall(r"\b[A-Z][a-zA-Z]+\b", question)
            if w.lower() not in _HEURISTIC_EXCLUDED_WORDS
        ]

        numeric_params = ("count", "number", "limit", "top")
        for param in template.parameters:
            if param in params:
                continue
            if param in numeric_params and numbers:
                params[param] = numbers.pop(0)
            elif param not in numeric_params and capitalized:
                params[param] = capitalized.pop(0)

        return params

    @staticmethod
    def _is_pattern_verified(
        template: QueryTemplate, param: str, value: str
    ) -> bool:
        """Whether ``value`` is exactly what the template's own pattern allows.

        Only ``template.parameter_patterns`` declares an intentional shape —
        NER and the heuristic fallback don't, so they never qualify here and
        always go through ``_sanitize_value()``. ``re.fullmatch`` is what
        "anchored" means in this context: it requires the pattern to account
        for the *entire* value, not just find a substring within it, so a
        pattern like ``\\b(\\d+)\\b`` still verifies correctly even without
        literal ``^``/``$`` (those would break extraction, which runs the
        same pattern against the whole question via ``re.findall``).
        """
        pattern = template.parameter_patterns.get(param)
        if not pattern:
            return False
        try:
            return re.fullmatch(pattern, value, re.IGNORECASE) is not None
        except re.error:
            logger.warning(
                "Invalid parameter_patterns regex for %r: %r", param, pattern
            )
            return False

    @staticmethod
    def _sanitize_value(value: str) -> str:
        """Remove characters that could cause injection.

        Allows: alphanumeric, spaces, hyphens, underscores.
        """
        return re.sub(r"[^\w\s-]", "", str(value))


def keyword_overlap_score(question: str, template_text: str) -> float:
    """Calculate keyword overlap between a question and a template.

    Removes stop words and template placeholders before comparison.

    Args:
        question: Normalized user question.
        template_text: Template pattern text.

    Returns:
        Float in [0.0, 1.0] representing the fraction of template
        keywords found in the question.
    """
    clean_template = re.sub(r"\{\w+\}", "", template_text)
    question_words = set(re.findall(r"\b\w+\b", question.lower()))
    template_words = set(re.findall(r"\b\w+\b", clean_template.lower()))

    question_words -= _STOP_WORDS
    template_words -= _STOP_WORDS

    if not template_words:
        return 0.0

    intersection = question_words & template_words
    return len(intersection) / len(template_words)
