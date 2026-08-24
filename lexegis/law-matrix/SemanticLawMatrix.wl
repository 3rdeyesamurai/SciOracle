(* ::Package:: *)

(* ==========================================================================
   Semantic Law Matrix — digital consumer law
   ==========================================================================

   A computable representation of the field: every classification of digital
   consumer law is a vector over normative primitives, so that "how close are
   these two bodies of law?" becomes a distance rather than an opinion.

   Three tensors:

     smat (class x primitive)       what a classification normatively demands
     cvg  (jurisdiction x class)    how strongly a jurisdiction occupies it
     cvg . sn                       the normative exposure of a jurisdiction,
                                    divided by its coverage and z-scored to give
                                    emphasis: what it weights, not how much it has

   Scales
     S entries  0 absent | 1 peripheral | 2 substantial | 3 defining
     C entries  0 none   | 1 partial/sectoral | 2 substantial | 3 comprehensive

   Provenance: the scores are an analyst scaffold prepared from the instruments
   cited per row, current to 2026-08. They are a research instrument, not legal
   advice, and every row should be re-checked against primary sources before it
   is relied on in a filing. `verify` on each row names what to re-check.

   Usage
     wolframscript -file SemanticLawMatrix.wl        (* writes the JSON *)
   or paste the file into a kernel and call SLMReport[].
   ========================================================================== *)

BeginPackage["SemanticLawMatrix`"];

SLMPrimitives::usage = "SLMPrimitives[] returns the normative primitive basis.";
SLMClasses::usage = "SLMClasses[] returns the classification rows.";
SLMJurisdictions::usage = "SLMJurisdictions[] returns the jurisdiction columns.";
SLMReport::usage = "SLMReport[] computes the full matrix analysis as an Association.";
SLMExport::usage = "SLMExport[path] writes the report to JSON.";
SLMPlots::usage = "SLMPlots[] returns the diagnostic plots.";

Begin["`Private`"];

(* -------------------------------------------------------------- basis --- *)
primitives = {
  <|"id" -> "consent",        "label" -> "Consent",              "gloss" -> "A freely given, specific act of permission is the gate to the conduct."|>,
  <|"id" -> "transparency",   "label" -> "Transparency",         "gloss" -> "Mandated disclosure of terms, identity, ranking, or provenance."|>,
  <|"id" -> "fairness",       "label" -> "Substantive fairness",  "gloss" -> "The balance of the bargain is policed, not merely its disclosure."|>,
  <|"id" -> "exit",           "label" -> "Exit and withdrawal",   "gloss" -> "The consumer may leave: cancel, withdraw, unsubscribe, port out."|>,
  <|"id" -> "redress",        "label" -> "Redress",              "gloss" -> "A remedy runs to the individual: repair, refund, damages, restitution."|>,
  <|"id" -> "liability",      "label" -> "Liability allocation",  "gloss" -> "The rule fixes who bears loss when the digital thing fails."|>,
  <|"id" -> "enforcement",    "label" -> "Enforcement severity",  "gloss" -> "Turnover-based fines, injunctions, or personal liability attach."|>,
  <|"id" -> "extraterritorial","label" -> "Extraterritorial reach","gloss" -> "The rule binds suppliers established outside the forum."|>,
  <|"id" -> "vulnerability",  "label" -> "Vulnerable users",      "gloss" -> "Heightened duties toward minors, the elderly, the distressed."|>,
  <|"id" -> "minimisation",   "label" -> "Data minimisation",     "gloss" -> "Purpose limitation and necessity constrain collection."|>,
  <|"id" -> "interoperability","label" -> "Interoperability",     "gloss" -> "Portability, open interfaces, switching without loss."|>,
  <|"id" -> "auditability",   "label" -> "Auditability",          "gloss" -> "Records, reporting, audits, or researcher access are compelled."|>,
  <|"id" -> "timeliness",     "label" -> "Timeliness",            "gloss" -> "Hard deadlines govern acting, responding, or notifying."|>,
  <|"id" -> "security",       "label" -> "Security duty",         "gloss" -> "A technical protection standard is imposed on the supplier."|>,
  <|"id" -> "prohibition",    "label" -> "Per se prohibition",    "gloss" -> "Conduct is banned outright, no balancing available."|>,
  <|"id" -> "exante",         "label" -> "Ex ante structure",     "gloss" -> "Compliance is designed in before launch, not litigated after."|>
};

primitiveIds = #["id"] & /@ primitives;

(* ------------------------------------------------------- jurisdictions --- *)
jurisdictions = {
  <|"id" -> "EU",     "label" -> "European Union",  "family" -> "civil / regulation-led"|>,
  <|"id" -> "UK",     "label" -> "United Kingdom",  "family" -> "common / regulator-led"|>,
  <|"id" -> "US_FED", "label" -> "United States (federal)", "family" -> "common / enforcement-led"|>,
  <|"id" -> "US_CA",  "label" -> "California",      "family" -> "common / statute-led"|>,
  <|"id" -> "CAN",    "label" -> "Canada",          "family" -> "mixed"|>,
  <|"id" -> "BR",     "label" -> "Brazil",          "family" -> "civil / code-led"|>,
  <|"id" -> "IN",     "label" -> "India",           "family" -> "common / statute-led"|>,
  <|"id" -> "CN",     "label" -> "China",           "family" -> "civil / administrative"|>,
  <|"id" -> "JP",     "label" -> "Japan",           "family" -> "civil"|>,
  <|"id" -> "KR",     "label" -> "South Korea",     "family" -> "civil"|>,
  <|"id" -> "AU",     "label" -> "Australia",       "family" -> "common / regulator-led"|>,
  <|"id" -> "SG",     "label" -> "Singapore",       "family" -> "common"|>
};

jurisdictionIds = #["id"] & /@ jurisdictions;

(* ----------------------------------------------------------- the rows --- *)
(* weights follow primitiveIds; coverage follows jurisdictionIds            *)
classRow[id_, label_, family_, weights_, coverage_, instruments_, verify_] :=
  <|"id" -> id, "label" -> label, "family" -> family,
    "weights" -> AssociationThread[primitiveIds -> weights],
    "coverage" -> AssociationThread[jurisdictionIds -> coverage],
    "instruments" -> instruments, "verify" -> verify|>;

classes = {
  classRow["data_protection", "General data protection", "Data and privacy",
    {3,3,2,1,2,2,3,3,2,3,2,3,2,2,1,2}, {3,3,1,3,2,3,3,3,3,3,2,3},
    {"EU GDPR 2016/679", "UK GDPR + DPA 2018", "CCPA/CPRA", "LGPD (BR)", "DPDP Act 2023 (IN)", "PIPL (CN)", "APPI (JP)"},
    "Confirm lawful-basis architecture and the current adequacy/transfer position for each forum."],

  classRow["eprivacy_tracking", "Cookies, trackers and communications confidentiality", "Data and privacy",
    {3,3,1,1,1,1,2,2,1,3,0,2,1,1,2,1}, {3,3,1,2,1,2,1,2,2,2,1,2},
    {"ePrivacy Directive 2002/58 art.5(3)", "PECR (UK)", "CIPA-based wiretap claims (US_CA)"},
    "ePrivacy remains a directive with divergent national consent rules; check the member-state rule, not the directive alone."],

  classRow["profiling_adtech", "Profiling and behavioural advertising", "Data and privacy",
    {3,3,2,1,1,1,2,2,3,3,0,2,1,1,2,2}, {3,2,1,3,1,2,2,2,2,2,1,2},
    {"GDPR arts.21-22", "DSA arts.26,28 (ad transparency, minors)", "CPRA opt-out of sharing", "FTC Act s.5 unfairness"},
    "Sensitive-category and minors advertising bans differ sharply; verify per platform tier."],

  classRow["portability_interop", "Data portability and interoperability", "Data and privacy",
    {1,2,1,2,1,1,1,2,0,1,3,2,2,2,0,3}, {3,2,1,2,2,2,2,1,1,2,1,1},
    {"GDPR art.20", "EU Data Act 2023/2854", "DMA arts.6(9),7", "CPRA s.1798.130"},
    "Data Act obligations phase in over time; confirm the applicable date and product scope."],

  classRow["automated_decisions", "Automated decision-making and consumer AI", "Data and privacy",
    {2,3,3,1,2,2,2,2,2,2,0,3,1,1,1,3}, {3,2,1,2,1,2,1,2,1,2,1,1},
    {"GDPR art.22", "EU AI Act 2024/1689", "Colorado AI Act", "CPRA ADMT regulations"},
    "AI Act duties phase in by risk tier; check which obligations are in force on the relevant date."],

  classRow["unfair_terms", "Unfair contract terms", "Contract and commerce",
    {1,2,3,2,3,2,2,1,2,0,0,0,1,0,2,0}, {3,3,1,2,3,3,2,2,2,2,3,2},
    {"Unfair Terms Directive 93/13", "Consumer Rights Act 2015 pt.2 (UK)", "CDC arts.51-54 (BR)", "ACL unfair contract terms"},
    "Australia moved to penalties for unfair terms in 2023; confirm the penalty exposure, not just voidness."],

  classRow["distance_selling", "Distance selling and pre-contractual information", "Contract and commerce",
    {1,3,2,2,2,1,2,1,1,0,0,1,3,0,1,0}, {3,3,1,1,2,3,2,3,2,3,2,2},
    {"Consumer Rights Directive 2011/83", "CCRs 2013 (UK)", "Act on Specified Commercial Transactions (JP)", "E-Commerce Rules 2020 (IN)"},
    "Information duties are itemised and jurisdiction-specific; compare item lists rather than assuming equivalence."],

  classRow["withdrawal_rights", "Cooling-off and right of withdrawal", "Contract and commerce",
    {1,2,2,3,2,1,2,1,1,0,0,1,3,0,1,0}, {3,3,0,1,2,3,2,3,2,3,1,1},
    {"CRD arts.9-16", "CDC art.49 (BR, 7 days)", "E-Commerce Law arts.49-50 (CN)"},
    "Digital-content withdrawal is lost on performance with consent; verify how that waiver is captured."],

  classRow["subscriptions_autorenewal", "Auto-renewal and cancellation", "Contract and commerce",
    {2,3,2,3,2,1,3,1,1,0,0,1,3,0,2,1}, {3,3,2,3,1,1,1,1,1,2,2,1},
    {"FTC Negative Option / click-to-cancel rule", "ROSCA (US)", "CA Automatic Renewal Law", "Omnibus Directive 2019/2161", "DMCC Act 2024 pt.4 (UK)"},
    "The FTC click-to-cancel rule has been litigated; confirm its status and effective date before relying on it."],

  classRow["pricing_transparency", "Price transparency and drip pricing", "Contract and commerce",
    {0,3,2,0,2,1,3,1,1,0,0,1,2,0,2,1}, {3,3,2,2,2,2,2,2,2,2,3,2},
    {"Price Indication Directive 98/6 + Omnibus", "FTC Rule on Unfair or Deceptive Fees", "DMCC Act 2024 (UK drip pricing)", "ACL component pricing"},
    "Junk-fee rules differ on whether taxes and shipping enter the headline price."],

  classRow["personalised_pricing", "Personalised and algorithmic pricing", "Contract and commerce",
    {1,3,3,0,1,1,2,1,2,2,0,2,1,0,1,1}, {2,2,0,1,0,1,1,1,1,1,1,0},
    {"CRD art.6(1)(ea) (personalised price disclosure)", "UCPD guidance on personalisation"},
    "Thin and mostly disclosure-based; surveillance-pricing enforcement is developing, so re-check activity."],

  classRow["digital_content_conformity", "Conformity of digital content and services", "Digital product quality",
    {0,2,2,2,3,3,2,1,1,0,1,1,2,1,0,0}, {3,3,0,1,1,2,1,2,1,1,2,1},
    {"Digital Content Directive 2019/770", "CRA 2015 pt.1 ch.3 (UK)", "ACL consumer guarantees"},
    "Objective conformity plus the update duty is an EU/UK construct; elsewhere it is general warranty law."],

  classRow["goods_digital_elements", "Goods with digital elements", "Digital product quality",
    {0,2,2,2,3,3,2,1,0,0,1,1,2,2,0,1}, {3,2,0,1,1,1,1,2,1,1,2,1},
    {"Sale of Goods Directive 2019/771", "Cyber Resilience Act 2024/2847"},
    "The CRA's obligations are phased; confirm which apply on the relevant date."],

  classRow["updates_repair", "Update duty, repair and obsolescence", "Digital product quality",
    {0,2,2,1,2,2,2,1,0,0,2,1,3,3,1,2}, {3,2,1,2,1,1,1,1,1,1,2,1},
    {"Directive 2024/1799 on repair", "Ecodesign / ESPR", "US state right-to-repair statutes"},
    "Support-period lengths and parts-availability duties vary by product category."],

  classRow["product_liability_digital", "Liability for defective digital products", "Digital product quality",
    {0,1,2,0,3,3,2,2,1,0,0,2,1,2,0,1}, {3,2,1,1,1,2,1,2,2,2,2,1},
    {"Product Liability Directive 2024/2853 (software in scope)", "state product liability doctrine (US)"},
    "The recast PLD brings software and AI into strict liability; confirm transposition status."],

  classRow["iot_security", "Consumer IoT and product cybersecurity", "Digital product quality",
    {0,2,1,0,1,2,2,2,1,1,1,2,2,3,2,3}, {3,3,1,2,1,1,1,2,1,2,1,2},
    {"Cyber Resilience Act", "RED delegated act 2022/30", "PSTI Act 2022 (UK)", "US Cyber Trust Mark", "CA SB-327"},
    "Default-password and vulnerability-disclosure duties are the common core; certification schemes differ."],

  classRow["platform_transparency", "Marketplace and ranking transparency", "Platforms and markets",
    {0,3,2,0,2,2,3,3,1,1,1,3,2,0,1,3}, {3,3,1,1,1,2,2,3,2,2,2,1},
    {"DSA arts.24-32", "P2B Regulation 2019/1150", "Omnibus ranking disclosure", "E-Commerce Law art.18 (CN)"},
    "Trader-identity (KYBC) duties apply only to marketplaces; check the service classification first."],

  classRow["intermediary_liability", "Intermediary liability and notice-and-action", "Platforms and markets",
    {0,2,2,1,2,3,2,3,1,0,0,2,3,1,1,2}, {3,3,2,1,1,2,2,3,2,2,2,2},
    {"DSA arts.4-9", "47 USC 230 (US)", "IT Rules 2021 (IN)", "Marco Civil art.19 (BR)"},
    "The US safe harbour and the EU due-diligence model point in opposite directions; do not merge them."],

  classRow["gatekeeper_obligations", "Gatekeeper and digital-markets duties", "Platforms and markets",
    {1,2,3,2,2,2,3,3,0,2,3,3,2,1,3,3}, {3,3,1,0,1,1,1,3,2,2,1,1},
    {"DMA 2022/1925", "DMCC Act 2024 pt.1 (UK SMS regime)", "AMPSA (JP)", "Platform rules (KR)"},
    "Designation is entity-specific; confirm whether the counterparty is actually designated."],

  classRow["content_moderation_rights", "User rights in content moderation", "Platforms and markets",
    {0,3,3,1,3,2,2,3,2,1,0,3,3,0,1,3}, {3,3,0,1,1,2,2,1,1,2,1,1},
    {"DSA arts.14-21 (statement of reasons, internal appeal, out-of-court settlement)", "Online Safety Act 2023 (UK)"},
    "Appeal and out-of-court dispute rights are largely an EU innovation; check local analogues carefully."],

  classRow["online_safety_illegal", "Illegal and harmful content duties", "Platforms and markets",
    {0,2,1,0,2,3,3,3,3,1,0,3,3,1,3,3}, {3,3,1,1,2,2,2,3,2,3,3,2},
    {"DSA", "Online Safety Act 2023 (UK)", "Online Safety Act 2021 (AU)", "IT Rules 2021 (IN)"},
    "Risk-assessment and age-assurance duties turn on service size and type."],

  classRow["unfair_commercial_practices", "Misleading and aggressive practices", "Conduct and manipulation",
    {0,3,3,1,3,2,3,2,3,0,0,1,1,0,3,0}, {3,3,3,3,3,3,3,2,3,3,3,3},
    {"UCPD 2005/29 + Annex I blacklist", "FTC Act s.5", "ACL ss.18,29", "CDC arts.36-38 (BR)"},
    "The nearest thing to a universal rule in this field; the blacklist contents still differ."],

  classRow["dark_patterns", "Manipulative interface design", "Conduct and manipulation",
    {3,3,3,2,2,1,3,2,3,2,0,2,1,0,3,1}, {3,3,2,3,1,1,1,2,1,2,2,1},
    {"DSA art.25", "UCPD guidance 2021", "CPRA s.1798.140(h) (consent obtained by dark pattern is not consent)", "FTC enforcement"},
    "Mostly enforced through existing unfairness law; only the DSA and CPRA name the conduct directly."],

  classRow["fake_reviews_influencers", "Fake reviews and undisclosed endorsements", "Conduct and manipulation",
    {0,3,2,0,2,2,3,2,2,0,0,2,1,0,3,1}, {3,3,3,2,2,2,2,2,2,2,3,2},
    {"Omnibus Directive review-verification duty", "FTC Rule on Consumer Reviews and Testimonials 2024", "DMCC Act 2024 (UK)"},
    "Verification duties (did the reviewer buy it?) are stronger than mere prohibition; check which applies."],

  classRow["direct_marketing_spam", "Unsolicited marketing and spam", "Conduct and manipulation",
    {3,2,1,3,2,1,3,2,2,2,0,2,2,0,3,0}, {3,3,3,2,3,2,2,2,2,3,2,3},
    {"ePrivacy art.13", "CAN-SPAM + TCPA (US)", "CASL (CAN)", "PDPA DNC (SG)"},
    "Opt-in versus opt-out is the fault line: CASL and ePrivacy are opt-in, CAN-SPAM is opt-out."],

  classRow["minors_protection", "Children, minors and age assurance", "Vulnerable users and access",
    {3,2,2,1,2,2,3,3,3,3,0,3,2,2,3,3}, {3,3,2,3,1,2,2,3,2,3,2,2},
    {"GDPR art.8", "COPPA (US)", "Age-Appropriate Design Code (UK, CA)", "DSA art.28", "Minors online protection rules (CN)"},
    "Age thresholds and assurance standards diverge; a compliant flow in one forum can be unlawful in another."],

  classRow["accessibility", "Accessibility of digital services", "Vulnerable users and access",
    {0,2,2,0,2,2,2,1,3,0,2,2,2,0,1,3}, {3,2,3,2,2,1,1,1,2,2,2,1},
    {"European Accessibility Act 2019/882", "ADA title III web litigation (US)", "s.508 (US)", "EN 301 549 / WCAG"},
    "The EAA applies from June 2025 with micro-enterprise carve-outs; confirm the entity size test."],

  classRow["payments_chargebacks", "Electronic payments and unauthorised transactions", "Finance and payments",
    {2,3,2,1,3,3,3,2,2,1,2,3,3,3,1,2}, {3,3,3,2,2,2,3,2,2,2,2,2},
    {"PSD2 2015/2366 (SCA, art.73 refunds)", "Reg E / EFTA (US)", "RBI authorised-payment framework (IN)"},
    "Liability-shift rules for unauthorised transactions differ in both cap and burden of proof."],

  classRow["crypto_consumer", "Crypto-asset consumer protection", "Finance and payments",
    {1,3,2,1,2,3,3,3,2,0,1,3,2,3,2,3}, {3,2,1,1,2,2,1,2,3,2,2,3},
    {"MiCA 2023/1114", "FCA financial promotions regime (UK)", "PSA amendments (JP)", "PS Act (SG)"},
    "Promotion rules and custody duties are moving fast; treat any score here as provisional."],

  classRow["credit_bnpl", "Digital credit and buy-now-pay-later", "Finance and payments",
    {1,3,3,2,3,2,3,1,3,1,0,2,3,1,2,2}, {3,3,2,2,2,2,2,2,2,2,3,2},
    {"Consumer Credit Directive 2023/2225 (BNPL in scope)", "CFPB BNPL interpretive rule (US)", "UK BNPL regulation"},
    "CCD2 applies from Nov 2026; confirm whether the transition rules cover the transaction."],

  classRow["redress_odr", "Collective redress and dispute resolution", "Enforcement",
    {0,2,2,1,3,1,3,2,2,0,0,2,3,0,0,1}, {3,2,1,1,2,3,2,2,1,2,2,1},
    {"Representative Actions Directive 2020/1828", "ADR Directive 2013/11", "class actions (US, AU)", "CDC collective actions (BR)"},
    "The EU ODR platform was wound down in 2025; do not cite it as a live route."],

  classRow["cross_border_enforcement", "Cross-border enforcement and cooperation", "Enforcement",
    {0,2,1,0,3,2,3,3,1,0,1,2,2,0,1,1}, {3,2,2,1,2,2,1,1,1,1,2,2},
    {"CPC Regulation 2017/2394", "ICPEN", "US SAFE WEB Act", "GPEN (privacy)"},
    "Cooperation instruments give regulators reach that private claimants do not have."]
};

classIds = #["id"] & /@ classes;

(* --------------------------------------------------------- the tensors --- *)
smat = Table[Lookup[c["weights"], primitiveIds], {c, classes}];          (* class x primitive *)
(* `C` is Protected in the kernel (constant of integration), hence cvg *)
cvg = Transpose[Table[Lookup[c["coverage"], jurisdictionIds], {c, classes}]]; (* jurisdiction x class *)

normalizeRow[v_] := With[{n = Norm[N[v]]}, If[n == 0, v, N[v]/n]];
sn = normalizeRow /@ smat;

(* --------------------------------------------------------- statistics --- *)
similarity = Table[sn[[i]] . sn[[j]], {i, Length[sn]}, {j, Length[sn]}];
distance = 1 - similarity;

density = Total /@ smat;                                   (* regulatory density of a class *)
reach = Total /@ Transpose[smat];                          (* how load-bearing a primitive is *)

(* Shannon entropy of each primitive's distribution across classes: a primitive
   spread evenly over the field is structural; a concentrated one is a specialism. *)
columnEntropy = Table[
   With[{col = N[Transpose[smat][[k]]]},
     With[{p = If[Total[col] == 0, col, col/Total[col]]},
       -Total[Select[p, # > 0 &] * Log[Select[p, # > 0 &]]] / Log[Length[col]]]],
   {k, Length[primitiveIds]}];

primitiveCorrelation = Correlation[N[smat]];

(* clustering over the normative geometry *)
clusterAssignments = With[{groups = FindClusters[
     Thread[sn -> classIds], 6, Method -> "Agglomerate", DistanceFunction -> CosineDistance]},
   Association @@ Flatten[MapIndexed[Function[{group, idx}, # -> First[idx] & /@ group], groups]]];

(* two-dimensional map by SVD of the centred normalised matrix *)
centred = # - Mean[sn] & /@ sn;
{uu, ss, vv} = SingularValueDecomposition[centred, 2];
embedding = uu . ss;
explained = With[{full = SingularValueList[centred]},
   Total[Take[full, 2]^2]/Total[full^2]];

(* Jurisdictions: exposure, emphasis, divergence and gaps.

   Cosine over the raw exposure sums is useless here -- every jurisdiction adds
   up the same 32 rows and so points in nearly the same direction (all pairwise
   similarities came back at 0.997-1.000). What distinguishes regimes is what
   they emphasise per unit of coverage, measured against the field consensus,
   so exposure is divided by total coverage and each primitive is z-scored
   across jurisdictions before any distance is taken. *)
exposure = cvg . sn;                                      (* jurisdiction x primitive *)
coverageScore = Total /@ cvg;
emphasis = Table[exposure[[j]]/coverageScore[[j]], {j, Length[jurisdictionIds]}];
emphasisZ = Transpose[Table[
    With[{col = Transpose[emphasis][[k]]}, (col - Mean[col])/StandardDeviation[col]],
    {k, Length[primitiveIds]}]];
jurisdictionDistance = Table[CosineDistance[emphasisZ[[a]], emphasisZ[[b]]],
   {a, Length[jurisdictionIds]}, {b, Length[jurisdictionIds]}];

(* What each regime over- and under-weights relative to the others. *)
signature = Association @@ Table[
   jurisdictionIds[[j]] -> <|
     "over" -> (primitiveIds[[#]] -> Round[emphasisZ[[j, #]], 0.01] & /@ Take[Ordering[-emphasisZ[[j]]], 3]),
     "under" -> (primitiveIds[[#]] -> Round[emphasisZ[[j, #]], 0.01] & /@ Take[Ordering[emphasisZ[[j]]], 3])|>,
   {j, Length[jurisdictionIds]}];

(* The pairs furthest apart in emphasis: where a single product design is most
   likely to satisfy one regulator and offend another. *)
widestPairs = Take[SortBy[Flatten[Table[
      <|"a" -> jurisdictionIds[[a]], "b" -> jurisdictionIds[[b]],
        "distance" -> Round[jurisdictionDistance[[a, b]], 0.001]|>,
      {a, Length[jurisdictionIds]}, {b, a + 1, Length[jurisdictionIds]}], 1], -#["distance"] &], 6];
gaps = Association @@ Table[
   jurisdictionIds[[j]] -> Take[
     SortBy[
       Select[Table[<|"class" -> classIds[[i]], "coverage" -> cvg[[j, i]], "density" -> density[[i]],
                      "exposure" -> cvg[[j, i]] * density[[i]]|>, {i, Length[classIds]}],
              #["coverage"] <= 1 &],
       -#["density"] &],
     UpTo[6]],
   {j, Length[jurisdictionIds]}];

(* Divergence risk: classifications where jurisdictions disagree most about how
   hard to regulate, weighted by how much normative load the class carries.
   These are the classes where a cross-border product is most likely to be
   lawful in one forum and unlawful in another. *)
divergence = Table[
   With[{col = N[Transpose[cvg][[i]]]},
     <|"class" -> classIds[[i]], "label" -> classes[[i]]["label"],
       "mean" -> Mean[col], "sd" -> StandardDeviation[col],
       "density" -> density[[i]],
       "risk" -> Round[StandardDeviation[col] * density[[i]] / 3, 0.01]|>],
   {i, Length[classIds]}];

(* Nearest neighbours in normative space: which bodies of law argue alike. *)
neighbours = Association @@ Table[
   classIds[[i]] -> Take[
     SortBy[Table[<|"class" -> classIds[[j]], "similarity" -> Round[similarity[[i, j]], 0.001]|>,
                  {j, Delete[Range[Length[classIds]], i]}], -#["similarity"] &],
     UpTo[4]],
   {i, Length[classIds]}];

(* ------------------------------------------------------------ report --- *)
SLMPrimitives[] := primitives;
SLMClasses[] := classes;
SLMJurisdictions[] := jurisdictions;

SLMReport[] := <|
  "generated" -> DateString[Now, "ISODate"],
  "scale" -> <|"weights" -> "0 absent | 1 peripheral | 2 substantial | 3 defining",
               "coverage" -> "0 none | 1 partial or sectoral | 2 substantial | 3 comprehensive"|>,
  "disclaimer" -> "Analyst scaffold current to 2026-08. Research instrument, not legal advice; \
re-check each row against primary sources before relying on it.",
  "primitives" -> primitives,
  "jurisdictions" -> jurisdictions,
  "classes" -> MapThread[
     Append[#1, <|"density" -> #2, "cluster" -> Lookup[clusterAssignments, #1["id"]],
                  "embedding" -> Round[#3, 0.001], "neighbours" -> Lookup[neighbours, #1["id"]]|>] &,
     {classes, density, embedding}],
  "matrix" -> <|"classIds" -> classIds, "primitiveIds" -> primitiveIds, "S" -> smat,
                "jurisdictionIds" -> jurisdictionIds, "C" -> cvg|>,
  "similarity" -> Round[similarity, 0.001],
  "primitiveStats" -> MapThread[
     Append[#1, <|"reach" -> #2, "entropy" -> Round[#3, 0.001]|>] &,
     {primitives, reach, columnEntropy}],
  "primitiveCorrelation" -> Round[primitiveCorrelation, 0.001],
  "exposure" -> Round[exposure, 0.001],
  "emphasisZ" -> Round[emphasisZ, 0.001],
  "jurisdictionDistance" -> Round[jurisdictionDistance, 0.001],
  "signature" -> signature,
  "widestPairs" -> widestPairs,
  "coverageScore" -> AssociationThread[jurisdictionIds -> coverageScore],
  "gaps" -> gaps,
  "divergence" -> SortBy[divergence, -#["risk"] &],
  "embeddingVariance" -> Round[explained, 0.001]
|>;

SLMExport[path_String] := Export[path, SLMReport[], "JSON", "Compact" -> False];

SLMPlots[] := <|
  "semantic" -> MatrixPlot[smat, ColorFunction -> "TemperatureMap",
     FrameTicks -> {{Thread[{Range[Length[classIds]], classIds}], None},
                    {None, Thread[{Range[Length[primitiveIds]], Rotate[#, Pi/2] & /@ primitiveIds}]}},
     ImageSize -> 900, PlotLabel -> "Classification x normative primitive"],
  "similarity" -> MatrixPlot[similarity, ColorFunction -> "SunsetColors", ImageSize -> 700,
     PlotLabel -> "Cosine similarity between classifications"],
  "coverage" -> MatrixPlot[cvg, ColorFunction -> "DeepSeaColors", ImageSize -> 900,
     FrameTicks -> {{Thread[{Range[Length[jurisdictionIds]], jurisdictionIds}], None}, {None, None}},
     PlotLabel -> "Jurisdiction x classification coverage"],
  "dendrogram" -> Dendrogram[Thread[sn -> classIds], DistanceFunction -> CosineDistance,
     ImageSize -> 900, PlotLabel -> "Agglomerative clustering over normative geometry"],
  "map" -> ListPlot[Callout[#[[1]], #[[2]]] & /@ Thread[{embedding, classIds}],
     ImageSize -> 800, PlotLabel -> "Two-dimensional map of the normative space",
     AspectRatio -> 0.7]
|>;

End[];
EndPackage[];

If[!TrueQ[$SLMNoAutoExport],
  SemanticLawMatrix`SLMExport[
    FileNameJoin[{DirectoryName[$InputFileName], "semantic-law-matrix.json"}]]];
