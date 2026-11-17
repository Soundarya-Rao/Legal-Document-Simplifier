from langchain_core.prompts import PromptTemplate

# Document category risk checklists tailored for Indian legal contexts
# Why: Different Indian contracts contain unique traps (e.g. bonds in employment, sub-limits in insurance).
# Supplying targeted risk checklists helps the model spot domain-specific unfair clauses reliably.
DOC_TYPE_CHECKLISTS = {
    "rental_lease": (
        "- Unfair security deposit deductions and delayed refund timelines (>7 days after vacating)\n"
        "- Lock-in periods & heavy penalties for early termination\n"
        "- Double or punitive rent damages (e.g. 2x rent) for holdover or delayed vacating\n"
        "- Arbitrary maintenance or elevator/generator running charges without sub-meter bills\n"
        "- Landlord inspection or entry without reasonable advance written notice\n"
        "- Forcing tenant to pay for structural, external, or preexisting plumbing/electrical repairs"
    ),
    "employment": (
        "- Post-employment non-compete clauses (largely void under Section 27 of Indian Contract Act, but often used to intimidate)\n"
        "- Unreasonable notice periods (>2-3 months) or forced forfeiture of earned salary in lieu of notice\n"
        "- Employment bonds, training cost recovery, or monetary penalties for resigning before a lock-in period\n"
        "- Unilateral power to reduce compensation, change designation, or enforce indefinite relocation\n"
        "- Overly broad intellectual property assignment covering personal projects created outside office hours without company tools\n"
        "- Immediate termination without written cause or statutory severance pay"
    ),
    "loan_credit": (
        "- Floating vs fixed interest rate traps and unilateral interest reset rights by lender\n"
        "- Excessive processing, prepayment, or foreclosure penalty charges\n"
        "- Penal compounding interest rates on delayed EMIs\n"
        "- Unlimited personal liability clauses for guarantors or co-borrowers\n"
        "- Irrevocable auto-debit (NACH/e-Mandate) consents without clear cancellation rights\n"
        "- Cross-collateralization and cross-default across unrelated financial facilities"
    ),
    "insurance": (
        "- Broad permanent exclusions and critical illness waiting periods (e.g. 2-4 year moratorium)\n"
        "- Restrictive room rent sub-limits and proportionate deduction penalties on overall hospital bills\n"
        "- Mandatory co-payment percentages borne by policyholder\n"
        "- Unrealistic claim intimation deadlines (e.g. mandatory notification within 24 hours of emergency hospitalization)\n"
        "- Arbitrary non-renewal, cancellation, or loss of continuity benefits upon policy portability"
    ),
    "property_sale": (
        "- Unclear title, unresolved encumbrances, or lack of clear indemnity for past property disputes\n"
        "- Asymmetric delayed possession penalties (e.g. builder pays 6% simple interest while buyer defaults at 18%)\n"
        "- Unilateral cancellation or excessive forfeiture of earnest money deposit (EMD)\n"
        "- Hidden escalation costs (clubhouse, electrification, floor rise, parking charges)\n"
        "- Absence of RERA registration guarantees or commitments to deliver Occupancy Certificate (OC)"
    ),
    "business_contract_nda": (
        "- Unlimited or one-sided indemnification clauses without reciprocal protections\n"
        "- Absence of mutual limitation of liability caps (e.g. uncapped consequential damages)\n"
        "- Unilateral termination for convenience without reasonable compensation for work completed\n"
        "- Expensive or foreign arbitration seats / jurisdictions (e.g. Singapore or London for domestic Indian disputes)\n"
        "- Indefinite or perpetual confidentiality obligations on standard commercial information"
    ),
    "terms_privacy": (
        "- Broad permissions to sell, monetize, or transfer personal data to third parties\n"
        "- Unilateral rights to alter terms, features, or fees without advance written notice\n"
        "- Automatic subscription renewal and recurring card debits with opaque cancellation paths\n"
        "- Forced arbitration clauses, jury waivers, or complete ban on class-action consumer claims\n"
        "- Arbitrary account suspension or termination with forfeiture of pre-paid balances or credits"
    ),
    "will_family": (
        "- Ambiguous asset descriptions or overlapping shares causing probate disputes\n"
        "- Unchecked executor powers lacking transparency or reporting obligations\n"
        "- Lack of alternate beneficiary provisions in case of premature demise\n"
        "- Missing explicit revocation of all previous wills and testamentary codicils"
    ),
    "legal_notice_court": (
        "- Unreasonably short response timelines (e.g. under 7-15 days)\n"
        "- Exaggerated, unsubstantiated damage claims or interest calculations\n"
        "- Threats of coercive criminal proceedings (e.g. Section 420 IPC / 138 NI Act) to settle purely civil disputes\n"
        "- Forced waivers of legal rights or statutory defenses"
    ),
    "other": (
        "- One-sided termination rights favoring only the drafting party\n"
        "- Unilateral rights to amend terms, pricing, or specifications without consent\n"
        "- Hidden fees, penalties, or automatic rollover terms\n"
        "- Broad indemnity obligations and absence of liability caps\n"
        "- Waiver of legal remedies, consumer court rights, or unreasonable jurisdiction"
    )
}

# Prompt to detect document type, category, and parties from the first ~3 pages
# Why: Classifying the document allows the analyzer to pull the correct risk checklist
# and tailor summaries specifically to that contract's nature.
DETECT_DOC_TYPE_PROMPT = PromptTemplate(
    input_variables=["document_sample"],
    template="""You are an expert Legal Document Classifier.
Analyze the opening pages of the legal document below and identify its document type, category, and parties involved.

CATEGORIES (You must select EXACTLY one of these identifiers):
- rental_lease
- employment
- loan_credit
- insurance
- property_sale
- business_contract_nda
- terms_privacy
- will_family
- legal_notice_court
- other

STRICT OUTPUT REQUIREMENT:
Return ONLY a valid JSON object. Do not include markdown code blocks, conversational text, intro, or outro.
Schema:
{{
  "doc_type": "Specific human-readable title (e.g. Residential Rental Agreement, Offer Letter, Term Life Insurance Policy)",
  "category": "one of the category keys above",
  "parties": ["Role 1 (e.g. Landlord/Owner)", "Role 2 (e.g. Tenant)"]
}}

DOCUMENT EXCERPT:
{document_sample}

JSON OBJECT:"""
)

# Generic prompt for full-document summarization across all Indian legal agreements
# Why: Works for any legal document (rental, job, loan, NDA) by structuring the summary
# into 7 clear sections, noting blank placeholders, and strictly forbidding hallucinations.
DOCUMENT_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["document_text", "language", "doc_type"],
    template="""You are an expert Legal Document Simplifier.
Your job is to read the entire legal document below ({doc_type}) and provide a clear, plain-language summary that an everyday person can easily understand.

INSTRUCTIONS:
Provide a well-structured summary in {language} covering these 7 sections:
1. What this document is: Clarify the exact purpose and legal nature of this document.
2. Who is involved and their roles: Identify all parties, their titles/roles, and who they represent.
3. Key dates and deadlines: Start date, duration, renewal dates, notice periods, and milestones.
4. Money involved: Amounts, recurring fees, deposits, interest rates, payment due dates, and penalties.
5. What each party must do: Core obligations and rights of each party.
6. What happens if something goes wrong: Breach conditions, penalties, dispute resolution, and termination rules.
7. Bottom line in 2-3 plain sentences: The practical takeaway for the user before signing.

STRICT RULES:
- Start directly with section 1. No introduction or closing remarks.
- Do not use horizontal rules or --- dividers anywhere.
- Only include a section if the document actually discusses it; if not covered, write "Not mentioned in the document".
- NEVER invent facts or make assumptions.
- If fields are blank or unfilled placeholders (e.g., "[Date]", "Rs. ___", "(Name)"), explicitly highlight that these details are blank.
- Use simple, everyday vocabulary with clear bullet points and bold section headers. Avoid complex legal jargon.

FULL LEGAL DOCUMENT:
{document_text}

SUMMARY (in {language}):"""
)

# Prompt for merging partial summaries of very long documents (> MAX_DIRECT_CHARS)
# Why: Large contracts processed in segmented page groups need a final unified synthesis
# that maintains consistent flow and eliminates duplication.
MERGE_SUMMARIES_PROMPT = PromptTemplate(
    input_variables=["partial_summaries", "language", "doc_type"],
    template="""You are an expert Legal Document Simplifier.
Below are section-by-section summaries from different parts of a long legal document ({doc_type}).
Synthesize them into one cohesive, comprehensive, and non-repetitive summary in {language}.

Start directly with section 1. No introduction or closing remarks.
Do not use horizontal rules or --- dividers anywhere.

STRUCTURE YOUR FINAL SUMMARY IN THESE 7 SECTIONS:
1. What this document is
2. Who is involved and their roles
3. Key dates and deadlines
4. Money involved (amounts, fees, interest, penalties)
5. What each party must do
6. What happens if something goes wrong (termination, breach, penalties)
7. Bottom line in 2-3 plain sentences

If any section was not mentioned anywhere in the document, state "Not mentioned in the document".
Highlight any blank placeholders (e.g. "[Date]", "Rs.___") found across the document.

PARTIAL SUMMARIES:
{partial_summaries}

FINAL UNIFIED SUMMARY (in {language}):"""
)

# Generic prompt for detecting risky clauses in any legal document
# Why: Evaluates document clauses against both domain-specific risk checklists
# and universal Indian contract pitfalls (unilateral cancellation, forced arbitration, vague fees).
RISKY_CLAUSES_PROMPT = PromptTemplate(
    input_variables=["document_text", "language", "doc_type", "checklist"],
    template="""You are an expert Legal Risk Analyst.
Review the entire legal document below ({doc_type}) and identify clauses that are potentially risky, one-sided, unfair, or restrictive.

DOMAIN-SPECIFIC CHECKLIST FOR {doc_type}:
{checklist}

UNIVERSAL TRAPS TO ALSO CHECK:
- One-sided termination rights
- Unilateral rights to amend terms, pricing, or duties without consent
- Hidden or vague fees, expenses, or cost transfers
- Auto-renewal terms with automatic financial commitments
- Waiver of legal remedies, statutory protections, or consumer court access
- Distant or expensive jurisdiction / arbitration seats
- Vague, ambiguous, or subjective obligations that can be exploited

STRICT OUTPUT REQUIREMENT:
Return ONLY a valid JSON array of objects. Do not include markdown code blocks, intro, or outro text.
Each object must follow this exact schema:
[
  {{
    "clause": "Short direct quote or clause title from the document",
    "page": 1,
    "risk_level": "High",
    "why_it_matters": "A simple explanation in {language} of why this clause could be unfavorable or risky for the user",
    "what_to_ask": "A specific practical question in {language} the user should ask the other party before signing"
  }}
]
Note: "risk_level" must strictly be one of: "High", "Medium", or "Low".

FULL LEGAL DOCUMENT:
{document_text}

JSON ARRAY:"""
)

# Generic RAG question-answering prompt for any Indian legal document
# Why: Operates on any contract type while enforcing grounding, page citations, and refusal to hallucinate.
LEGAL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question", "language"],
    template="""You are an expert Legal Document Assistant.
Your job is to answer questions about the legal document clearly, accurately, and in plain language that anyone can understand.

STRICT INSTRUCTIONS:
1. Answer the question relying ONLY on the facts directly provided in the CONTEXT below.
2. Do NOT guess, extrapolate, or bring in outside information not mentioned in the CONTEXT.
3. If the answer is NOT present in the CONTEXT, you must explicitly state: "This information is not mentioned in the provided document." Do not try to make up an answer.
4. Always cite the exact page number(s) from which you obtained the answer (e.g., "[Page 1]", "[Page 3]").
5. Respond completely in the requested LANGUAGE: {language}. Use natural, respectful, and easily understandable language.

CONTEXT FROM LEGAL DOCUMENT:
{context}

PREVIOUS CONVERSATION HISTORY:
{chat_history}

USER QUESTION:
{question}

ANSWER (in {language}, citing page numbers):"""
)
