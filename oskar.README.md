# Implementasjonsplan for TEK17 RAG og refusal-analyse

Denne planen beskriver steg for steg hvordan vi bygger en RAG-pipeline for TEK17 med fokus på kontroll over embedding, retrieval, kontekst og refusals. Rekkefølgen er bevisst lagt opp slik at vi først får et godt eval-sett og tydelig scope, og deretter bygger teknisk infrastruktur rundt det.

## 1. Scope og eval-sett (før noe RAG)

1. **Velg kapitler/paragrafer vi dekker i første versjon**  
	 - Lag f.eks. `docs/scope.md` med liste over kapitler/§§ vi faktisk skal støtte nå.

2. **Lag første eval-sett for disse kapitlene**  
	 - Fil: `analysis/questions/tek17_eval_questions.jsonl`.  
	 - For hvert spørsmål:
		 - `id`
		 - `question`
		 - `target_sections`: liste over `section_id`/paragrafer som *burde* brukes
		 - `difficulty`: f.eks. `easy`/`medium`/`hard`
		 - `should_refuse`: `true/false` – om vi forventer refusal eller ikke

	 Eksempel:

	 ```json
	 {
		 "id": "q1",
		 "question": "Hva er formålet med § 1-1?",
		 "target_sections": ["§ 1-1"],
		 "difficulty": "easy",
		 "should_refuse": false
	 }
	 ```

	 Dette eval-settet blir sannhetskilden for:
	 - Retrieval-kvalitet (fant vi riktige paragrafer?)
	 - Refusal-kvalitet (nektet modellen når den ikke skulle, eller omvendt)

## 2. Embedding-strategi og lagring

3. **Definér representasjon for TEK17-chunks**  
	 - Fil: `data/processed/tek17_chunks.jsonl`.  
	 - Felter per rad, f.eks.:
		 - `chunk_id`
		 - `section_id`
		 - `chapter`
		 - `title`
		 - `text`
		 - `source_url`
		 - `chunk_type` (f.eks. `reg` / `guidance`)
		 - `eval_expected_for`: liste av eval-spørsmål (question-id) denne chunken *burde* være relevant for

	 Viktig: `eval_expected_for` er **kun metadata for analyse**. Retrieval bruker det ikke som filter; eval-skriptene bruker det i etterkant for å se om RAG har hentet «riktige» chunks.

4. **Velg embedding-modell og vektor-database**  
	 - Dokumentér i `docs/embeddings.md`:
		 - Modellnavn + provider (f.eks. OpenAI, lokal modell via Ollama, etc.)
		 - Dimensjon
		 - Språkstøtte (norsk)
	 - Velg vektor-lagring:
		 - Lokalt: f.eks. Postgres + pgvector, Qdrant
		 - Hosted: Pinecone, Weaviate, osv.

5. **Implementer embedding-pipeline**  
	 - Script som:
		 - Leser `tek17_chunks.jsonl`
		 - Beregner embedding for `text`
		 - Lagrer i vektor-databasen med komplett metadata (inkl. `eval_expected_for`).

## 3. Retrieval-design

6. **Definér internt retrieval-API**  
	 - Eget Python-interface, f.eks. i `src/tek17/rag/retrieval.py`:
		 - `search(query: str, k: int = 10, filters: dict | None = None) -> list[Result]`
	 - `Result` bør inneholde:
		 - `chunk_id`, `section_id`, `score`, `text`, `metadata`.

7. **Velg grunnleggende retrieval-strategi**  
	 - Start enkelt:
		 - Ren vektorsøk på embedding + eventuelt filtrering på chapter/`chunk_type`.
	 - Logg alltid:
		 - Brukerspørsmål
		 - Top-k treff (id, score, kort tekstutdrag)

8. **(Senere) legg til re-ranking hvis nødvendig**  
	 - F.eks. en egen re-ranker som ser på matching mellom query og tekst, eller LLM-basert re-ranking.

## 4. Modell-lag: byttbart mellom Ollama, OpenAI, Gemini

9. **Lag et felles interface for LLM-kall**  
	 - Egen modul, f.eks. `src/tek17/rag/llm_client.py` med funksjon:
		 - `generate(messages, provider: str, model: str, **kwargs) -> str`
	 - Implementer adaptere for:
		 - `ollama`
		 - `openai`
		 - `gemini`
	 - Konfigurasjon via f.eks. `config.yaml` eller `pyproject.toml`:
		 - Provider, modellnavn, temperatur, max_tokens, osv.

10. **Standardiser konteksten modellen får**  
		- Meldingsformat:
			- `system`: rollebeskrivelse – f.eks. at modellen kun skal bruke TEK17-tekst i konteksten, gi forklaringer, ikke konkret juridisk rådgivning.
			- `user`: selve spørsmålet.
			- En eksplisitt «kontekst»-seksjon med de retrieved chunks:
				- Nummererte utdrag: `[DOC 1] ...`, `[DOC 2] ...` osv., med `section_id` og tittel.
		- Logg før kall:
			- Provider og modellnavn
			- Systemprompt
			- User-tekst
			- Kontekst-tekst (hvilke chunks, med metadata)

		Dette gjør at vi *alltid* kan se hvilken kontekst modellen hadde da den eventuelt refuserte.

## 5. RAG-orchestrator

11. **Bygg en samlet RAG-funksjon**  
		- F.eks. `src/tek17/rag/orchestrator.py` med funksjon:
			- `answer_question(question: str, provider: str, model: str, debug: bool = False) -> dict`
		- Steg i funksjonen:
			1. Kjør `retriever.search(question, k=...)`.
			2. Bygg meldinger (`system`, `user`, og kontekst med retrieved chunks).
			3. Logg full prompt + metadata.
			4. Kall `llm_client.generate(...)`.
			5. Returner strukturert resultat:
				 - `answer`
				 - `retrieved_chunks`
				 - `raw_prompt`
				 - `provider`
				 - `model`
				 - `did_refuse` (simple heuristikk til å begynne med)

12. **Støtt "debug-modus"**  
		- Ved `debug=True` skal funksjonen også returnere:
			- Hele konteksten som ble sendt til modellen
			- Alle relevante metadata per chunk
		- Dette gjør det enkelt å se hvor i kjeden ting går galt (embedding, retrieval, prompt, modell).

## 6. Eval-strategier (retrieval + refusals)

13. **Evaluer retrieval isolert (før LLM)**  
		- Eval-script, f.eks. `src/tek17/rag/eval_retrieval.py` som:
			- Leser `analysis/questions/tek17_eval_questions.jsonl`.
			- For hvert spørsmål:
				- Kjører `search(question, k=K)`.
				- Sammenligner `section_id` i resultatene med `target_sections` fra eval-settet.
			- Rapporterer f.eks.:
				- Recall@K: andel spørsmål der minst én `target_section` er i top-K.
		- Bruk i tillegg `eval_expected_for`-feltet på chunks som sanity-sjekk (f.eks. hvor mange ganger et spørsmål faktisk får de chunkene vi forventer).

14. **Evaluer full RAG (inkludert refusals)**  
		- Eget script, f.eks. `src/tek17/rag/eval_rag.py` som:
			- Leser `tek17_eval_questions.jsonl`.
			- Kjører hele RAG-pipen (retrieval + LLM) for hver `question`.
			- Lagrer for hvert spørsmål:
				- `answer`
				- `retrieved_section_ids`
				- `did_refuse` (heuristikk: regex på fraser som «kan ikke svare», «kan ikke gi juridisk rådgivning» osv.)
				- `should_refuse` (fra eval-settet)
			- Bygger en forvirringsmatrise for refusals:
				- True Accept (skulle svare, svarte)
				- False Refusal (skulle svare, nektet)
				- True Refusal (skulle nekte, nektet)
				- False Accept (skulle nekte, svarte)

15. **Koble retrieval-feil og refusal-feil**  
		- For hvert eval-spørsmål:
			- Sjekk om minst én `target_section` var med blant `retrieved_section_ids`.
			- Kombinér med refusal-info:
				- Hvis riktig paragraf er hentet, men modellen refuserer → sannsynlig prompt/safety-problem.
				- Hvis riktige paragrafer *ikke* er hentet → embedding/chunking/retrieval-problem.

## 7. Iterasjon på prompt og refusjonslogikk

16. **Tydelig systemprompt for refusjoner**  
		- Spesifiser eksplisitt at modellen:
			- Skal forklare innholdet i TEK17 og veiledningen.
			- Ikke skal gi individuelle, bindende juridiske råd, bare forklare regeltekst og veiledning.
			- Må begrunne refusjoner («jeg kan ikke svare fordi …» og hva som mangler i konteksten).

17. **Løpende eksperimenter og logging**  
		- For hver endring i systemprompt eller innstillinger:
			- Kjør eval-skriptene på nytt.
			- Lag en kort logg i `docs/refusal_experiments.md` med:
				- Prompt-versjon
				- Modell/provider
				- Nøkkel-metrikker (Recall@K, andel feil refusals osv.)

## 8. UI og modulær arkitektur

18. **Enkel UI for manuell testing**  
		- For eksempel:
			- CLI-kommando i `tek17`-pakken som lar deg skrive inn spørsmål og som viser:
				- Svar
				- Hvilke paragrafer/chunks som ble brukt
				- Om modellen refuserte eller ikke.
			- Alternativt en liten web-UI (Streamlit/FastAPI) som du allerede er i gang med.

19. **Sørg for modulær oppbygning**  
		- Foreslått struktur:
			- `src/tek17/rag/` – orchestrator, eval, klienter
			- `src/tek17/rag/retrieval.py` – all retrieval-logikk
			- `src/tek17/rag/llm_client.py` – adaptere for Ollama/OpenAI/Gemini
			- `analysis/questions/` – eval-spørsmål og eval-resultater
		- Poeng: lett å bytte LLM-provider uten å endre retrieval-kode, og lett å bytte embedding-strategi uten å røre eval-settet.

---

Neste naturlige steg er å faktisk opprette `analysis/questions/tek17_eval_questions.jsonl` og en første versjon av `data/processed/tek17_chunks.jsonl`, og deretter begynne på retrieval- og eval-skriptene i `src/tek17/rag/`.

