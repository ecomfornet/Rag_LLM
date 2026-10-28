"""
Generation Module - OpenAI GPT-4o-mini
LLM generation with chain-of-thought and validation
"""
import logging
from typing import Dict, List
from openai import OpenAI
import openai
import re

import config

logger = logging.getLogger(__name__)


class LLMGenerator:
    """Generates responses using OpenAI GPT-4o-mini"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=config.OPENAI_API_KEY,
            timeout=60.0,   # LLM completions may take longer; still prevents infinite hangs
        )
        logger.info("LLMGenerator initialized")
    
    # ------------------------------------------------------------------
    # Helper: extract a metadata field from a chunk dict.
    #
    # The retrieval pipeline (retrieval_4.py) stores metadata **flat** on
    # the chunk dict (e.g. chunk['doc_title']).  The test harness in main()
    # uses a nested format (chunk['metadata']['doc_title']).  This helper
    # transparently supports both shapes so neither path breaks.
    # ------------------------------------------------------------------
    @staticmethod
    def _get_meta(chunk: Dict, key: str, default=''):
        """Return chunk[key], or chunk['metadata'][key] if it exists there."""
        # Flat format (real pipeline)
        if key in chunk:
            return chunk[key]
        # Nested format (test harness / legacy)
        m = chunk.get('metadata', {})
        return m.get(key, default)

    def build_legal_prompt(self, query: str, context: str, chunks: List[Dict] = None) -> str:
        """
        Build chain-of-thought prompt for legal reasoning.
        When chunks are provided, a structured source table is injected so the
        LLM can reason about provenance (institution, reference, year, date).
        
        Args:
            query:   User query
            context: Retrieved context string (from retrieve_with_context)
            chunks:  Optional list of chunk dicts carrying full metadata
            
        Returns:
            Formatted prompt
        """
        # Build a compact source provenance table from chunk metadata
        sources_block = ""
        if chunks:
            rows = []
            seen = set()
            for chunk in chunks:
                # FIX: real pipeline uses 'id' OR 'doc_id'; test harness uses 'doc_id'
                doc_id = chunk.get('doc_id') or chunk.get('id', '')
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                parts = []
                # FIX: use _get_meta to support both flat and nested chunk shapes
                doc_title = self._get_meta(chunk, 'doc_title') or self._get_meta(chunk, 'title')
                if doc_title:                                 parts.append(f"Titre: {doc_title}")
                if self._get_meta(chunk, 'type'):             parts.append(f"Type: {self._get_meta(chunk, 'type')}")
                if self._get_meta(chunk, 'reference'):        parts.append(f"Réf: {self._get_meta(chunk, 'reference')}")
                # IMPORTANT: Use the legal year extracted from title/reference,
                # not the WordPress publish/modified date, as temporal anchor.
                if self._get_meta(chunk, 'year'):             parts.append(f"Année: {self._get_meta(chunk, 'year')}")
                if self._get_meta(chunk, 'institution'):      parts.append(f"Institution: {self._get_meta(chunk, 'institution')}")
                raw_url = self._get_meta(chunk, 'source_url')
                if raw_url:                                   parts.append(f"URL: {raw_url}")
                if parts:
                    rows.append("  • " + " | ".join(parts))
            if rows:
                sources_block = "\n# SOURCES IDENTIFIÉES (Utilisez impérativement ces informations pour vos citations)\n" + "\n".join(rows) + "\n"

        prompt = f"""# RÔLE
Vous êtes un expert juridique spécialisé en droit OHADA et droit guinéen.
{sources_block}
# CONTEXTE JURIDIQUE
{context}

# QUESTION
{query}

# RAISONNEMENT REQUIS
Avant de répondre, analysez étape par étape:

1. **Identification des textes applicables:**
   - Quelles LOI(S), DÉCRET(S), ARRÊTÉ(S) s'appliquent? Citer références complètes (ex: Loi L/2017/056/AN, Décret D/2018/239).
   - Hiérarchie: Loi > Décret > Arrêté. Institution émettrice et date pour chaque texte.

2. **Valeurs numériques OBLIGATOIRES:**
   - TAUX (%): impôts, cotisations — chiffres exacts.
   - MONTANTS: plafonds, seuils, amendes.
   - DÉLAIS: jours, mois pour formalités.
   - Renvois inter-articles: extraire la valeur finale (ex: article 229 = 15%).

3. **Analyse chronologique:**
   - Texte en vigueur? Modifications/abrogations?

4. **Synthèse:**
   - Réponse détaillée, structurée par type (Loi, Décret, Arrêté).
   - TOUTES les valeurs numériques du contexte doivent apparaître. Ne jamais renvoyer à un article sans donner la valeur si présente dans le contexte.
   - TRÈS IMPORTANT: NE PAS INVENTER de loi, décret, article ou date qui ne figure pas explicitement dans le CONTEXTE ou les SOURCES IDENTIFIÉES. Si l'information demandée n'apparaît pas dans les documents disponibles, vous devez le dire clairement.

# FORMAT RÉPONSE

**RÉPONSE:**
[Réponse DÉTAILLÉE. Commencer par les faits concrets: taux exacts, montants, délais. Structurer par texte (Loi X, Décret Y). Inclure TOUTES les valeurs numériques pertinentes. Citer article par article si nécessaire.
SI l'information exacte demandée (par exemple les dispositions détaillées d'un ancien code, ou le contenu d'une décision) N'APPARAÎT PAS dans le contexte, vous devez l'indiquer explicitement, par exemple: "Les dispositions précises de ce texte ne figurent pas dans la base documentaire actuelle; la réponse ci-dessous est limitée à ce que contiennent les documents disponibles." Ne jamais prétendre que le texte est dans la base si le CONTEXTE ne le montre pas.]

**FONDEMENT JURIDIQUE:**
Pour CHAQUE source, fournir une ligne structurée:
- **Loi/Convention:** [Réf] — [Institution] — [Date] — [Articles cités] — [URL]
- **Décret:** [Réf] — [Objet] — [Articles] — [URL]
- **Arrêté:** [Réf] — [Objet] — [URL]
IMPORTANT: Séparer textes OHADA et textes guinéens. Citer Loi ET son Décret d'application si présents.
Ne citer comme fondement juridique QUE les textes qui apparaissent réellement dans le CONTEXTE ou les SOURCES IDENTIFIÉES. Si aucun texte précis n'est disponible pour répondre entièrement à la question, l'indiquer explicitement (par exemple: "Les références exactes aux dispositions antérieures ne figurent pas dans les documents indexés").

**CONFIANCE:**
[Score entre 0.0 et 1.0 évaluant la certitude juridique de la réponse basée uniquement sur les sources fournies]
"""

        return prompt
    
    def generate_response(self, query: str, context: str, chunks: List[Dict] = None) -> Dict:
        """
        Generate response using OpenAI.
        When chunks are provided, metadata is injected into the prompt and
        returned in the result dict as 'sources'.

        The 'sources' list uses both 'title' AND 'doc_title' keys so the
        HTML frontend (which reads src.title) and any backend code that
        reads src.doc_title both work without changes.
        
        Args:
            query:   User query
            context: Retrieved context string
            chunks:  Optional list of chunk dicts with full metadata
            
        Returns:
            Response dictionary including 'sources' list
        """
        # Build prompt (with metadata table if chunks available)
        prompt = self.build_legal_prompt(query, context, chunks=chunks)
        
        # Call OpenAI
        try:
            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Vous êtes un expert juridique OHADA/Guinée."},
                    {"role": "user", "content": prompt}
                ],
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                top_p=config.LLM_TOP_P
            )
            
            answer = response.choices[0].message.content
            usage = {
                'prompt_tokens':      response.usage.prompt_tokens,
                'completion_tokens':  response.usage.completion_tokens,
                'total_tokens':       response.usage.total_tokens
            }
            
            logger.info(f"Generated response: {usage['total_tokens']} tokens")
            
            # Build deduplicated sources list from chunk metadata.
            # FIX: use _get_meta() so that both real (flat) and test (nested)
            # chunk shapes are handled correctly, preventing an empty sources list.
            sources = []
            if chunks:
                seen = set()
                for chunk in chunks:
                    # FIX: real pipeline uses 'id'; test harness uses 'doc_id'
                    doc_id = chunk.get('doc_id') or chunk.get('id', '')
                    if doc_id in seen:
                        continue
                    seen.add(doc_id)
                    # Resolve doc_title from either flat or nested metadata
                    doc_title = (
                        self._get_meta(chunk, 'doc_title')
                        or self._get_meta(chunk, 'title')
                    )
                    sources.append({
                        # 'title' used by app.py frontend (src.title)
                        'title':       doc_title,
                        # 'doc_title' kept for internal Python callers
                        'doc_title':   doc_title,
                        'doc_id':      doc_id,
                        'type':        self._get_meta(chunk, 'type'),
                        'reference':   self._get_meta(chunk, 'reference'),
                        'year':        self._get_meta(chunk, 'year'),
                        'institution': self._get_meta(chunk, 'institution'),
                        'date':        self._get_meta(chunk, 'date'),
                        'modified':    self._get_meta(chunk, 'modified'),
                        'source_url':  self._get_meta(chunk, 'source_url'),
                        # Retrieval score for UI display
                        'score':       chunk.get('rrf_score', chunk.get('score', 0)),
                        'rrf_score':   chunk.get('rrf_score'),
                        # Source type badge (vector / graph_expansion)
                        'source':      chunk.get('source', 'vector'),
                        # Content preview for expandable card
                        'content':     chunk.get('content', ''),
                    })
            
            return {
                'answer':  answer,
                'model':   config.LLM_MODEL,
                'usage':   usage,
                'sources': sources,
            }
            
        except openai.RateLimitError:
            logger.error(
                "OpenAI rate limit (429) — retries exhausted. "
                "Consider raising max_retries or upgrading your tier."
            )
            raise
        except openai.AuthenticationError:
            logger.error("OpenAI authentication failed — check OPENAI_API_KEY in .env")
            raise
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI network error: {e.__cause__}")
            raise
        except openai.APIStatusError as e:
            logger.error(f"OpenAI API error {e.status_code}: {e.message}")
            raise
    
    def compute_cost(self, usage: Dict) -> float:
        """
        Compute cost of API call.
        Pricing is read from config.py (LLM_INPUT_COST_PER_1M / LLM_OUTPUT_COST_PER_1M)
        so a model or pricing change only requires a single edit in config.py.
        
        Args:
            usage: Usage dictionary
            
        Returns:
            Cost in USD
        """
        input_cost  = (usage['prompt_tokens']      / 1_000_000) * config.LLM_INPUT_COST_PER_1M
        output_cost = (usage['completion_tokens']  / 1_000_000) * config.LLM_OUTPUT_COST_PER_1M
        return input_cost + output_cost


class ResponseFormatter:
    """Formats and validates responses"""
    
    @staticmethod
    def extract_sections(response: str) -> Dict:
        """Extract structured sections from response"""
        sections = {}
        if not response or not isinstance(response, str):
            return sections

        # 1. Extract main response
        response_match = re.search(
            r'(?:(?:\*\*|#{1,6}\s*|\*)|(?:^|\r?\n)[ \t]*)R[EÉ]PONSE\b\*{0,2}[ \t]*:?[ \t]*\*{0,2}[ \t]*(?:\r?\n)?\s*'
            r'(.*?)'
            r'(?=(?:\r?\n|[ \t]+(?=\*\*|#{1,6}))[ \t]*(?:'
            r'(?:\*\*|#{1,6}\s*|\*)?(?:FONDEMENT(?:\s+JURIDIQUE)?|CONFIANCE)\b(?:\*{0,2}[ \t]*:|\*{1,2}[ \t]*(?:\r?\n|[-+0-9\[]))'
            r'|(?:FONDEMENT(?:\s+JURIDIQUE)?|CONFIANCE)\b[ \t]*:'
            r')|\Z)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if response_match:
            sections['response'] = response_match.group(1).strip()

        # 2. Extract legal basis cleanly up to CONFIANCE header or end-of-string
        basis_match = re.search(
            r'(?:(?:\*\*|#{1,6}\s*|\*)|(?:^|\r?\n)[ \t]*)FONDEMENT(?:\s+JURIDIQUE)?\b\*{0,2}[ \t]*:?[ \t]*\*{0,2}[ \t]*(?:\r?\n)?\s*'
            r'(.*?)'
            r'(?=(?:\r?\n|[ \t]+(?=\*\*|#{1,6}))[ \t]*(?:'
            r'(?:\*\*|#{1,6}\s*|\*)?CONFIANCE\b(?:\*{0,2}[ \t]*:|\*{1,2}[ \t]*(?:\r?\n|[-+0-9\[]))'
            r'|CONFIANCE\b[ \t]*:'
            r')|\Z)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if basis_match:
            sections['legal_basis'] = basis_match.group(1).strip()

        # 3. Extract confidence score with brackets, percentages, negatives, fractions, and robust 0.5 fallback
        confidence_match = re.search(
            r'(?:(?:\*\*|#{1,6}\s*|\*)|(?:^|\r?\n)[ \t]*)CONFIANCE\b'
            r'(?:\*{0,2}[ \t]*:[ \t]*\*{0,2}|\*{1,2}[ \t]*(?:\r?\n|$))'
            r'[ \t\r\n]*\[?\*?\*?\s*([^\s\n()\]*]+(?:[ \t]*%)?)',
            response,
            re.IGNORECASE
        )
        if confidence_match:
            val = confidence_match.group(1).strip().strip("[]*")
            is_pct = val.endswith('%')
            clean_val = val.rstrip('%').strip().replace(',', '.')

            if '/' in clean_val:
                parts = clean_val.split('/')
                try:
                    score = float(parts[0]) / float(parts[1])
                except Exception:
                    score = 0.5
            else:
                try:
                    score = float(clean_val)
                    import math
                    if math.isnan(score) or math.isinf(score):
                        score = 0.5
                    elif is_pct:
                        score = score / 100.0
                except ValueError:
                    score = 0.5

            sections['confidence'] = min(1.0, max(0.0, score))

        return sections
    
    @staticmethod
    def validate_response(response: str, context: str) -> tuple:
        """
        Validate response quality.
        Also checks for institution and reference citations
        that come from the new metadata structure.
        
        Args:
            response: Generated response
            context:  Source context
            
        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []
        
        # Check minimum length
        if len(response) < 100:
            issues.append("Response too short")
        
        # Check for legal citations (extended to include reference codes and institutions)
        citation_pattern = r'(article|loi|décret|code|arrêté|ordonnance|jugement|arrêt|[A-Z]/\d{4}/\d+|N°\s*\d+)'
        if not re.search(citation_pattern, response, re.IGNORECASE):
            issues.append("No legal citations found")
        
        # Check for confidence score in LLM answer text.
        # In agentic mode the confidence is computed separately by
        # LegalReasonerAgent and injected into metadata — not in the text.
        # Downgrade to INFO so it doesn't trigger a validation failure.
        if not re.search(r'confiance', response, re.IGNORECASE):
            issues.append("[INFO] Confidence score not found in answer text")
        
        # Check for uncertainty indicators (informational only — not a validation failure).
        # These are HONEST model responses when the context lacks the answer;
        # flagging them as hallucinations would be incorrect.
        uncertainty_phrases = [
            "je ne sais pas",
            "information non disponible",
            "contexte insuffisant"
        ]
        for phrase in uncertainty_phrases:
            if phrase.lower() in response.lower():
                # Append as INFO — does NOT affect is_valid
                issues.append(f"[INFO] Uncertainty detected: '{phrase}'")
        
        # is_valid ignores INFO-level notices
        is_valid = not any(
            i for i in issues if not i.startswith("[INFO]")
        )
        
        return is_valid, issues


def main():
    """Test generation"""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT
    )
    
    generator = LLMGenerator()
    formatter = ResponseFormatter()
    
    # Test query with context in the new retrieve_with_context() format
    query = "Quel est le capital minimum d'une SARL?"
    
    # Simulate the new context format produced by retrieval_4.py
    context = """
[Document 1] Loi sur les SARL | type=loi | réf=L/2023/015 | année=2023 | inst=Présidence | date=2023-04-01
Article 15: Le capital social minimum d'une SARL est fixé à 1.000.000 GNF.
Les apports peuvent être en numéraire ou en nature.
Source: https://kytisa.com/loi-sarl-2023
    """
    
    # Simulate chunk list with full metadata
    mock_chunks = [{
        'doc_id': '999',
        'source': 'vector',
        'score':  0.92,
        'content': 'Article 15: Le capital social minimum d\'une SARL est fixé à 1.000.000 GNF.',
        'metadata': {
            'doc_title':   'Loi sur les SARL',
            'type':        'loi',
            'reference':   'L/2023/015',
            'year':        2023,
            'institution': 'Présidence',
            'date':        '2023-04-01T00:00:00',
            'modified':    '2025-01-10T00:00:00',
            'source_url':  'https://kytisa.com/loi-sarl-2023',
        }
    }]
    
    print(f"Query: {query}\n")
    
    result = generator.generate_response(query, context, chunks=mock_chunks)
    
    print(f"Response:\n{result['answer']}\n")
    print(f"Model:  {result['model']}")
    print(f"Tokens: {result['usage']['total_tokens']}")
    print(f"Cost:   ${generator.compute_cost(result['usage']):.6f}")
    
    # Print sources — verify both 'title' and 'doc_title' are present
    print(f"\nSources ({len(result['sources'])}):")
    for src in result['sources']:
        print(f"  • title={src['title']} | doc_title={src['doc_title']} | "
              f"{src['type']} | {src['reference']} | {src['year']} | "
              f"{src['institution']} | source={src['source']} | score={src['score']}")
    
    # Validate
    is_valid, issues = formatter.validate_response(result['answer'], context)
    print(f"\nValidation: {'✓ Valid' if is_valid else '✗ Issues found'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")


if __name__ == "__main__":
    main()