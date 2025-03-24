Title: How to Calculate Persona Weights (Simplified)

Steps:
- Base Weight: Take plan feature (e.g., ma_drug_coverage), cap at 0.5 (0.7 for CSNP), boost if CSNP/DSNP type = "Y".
- Behavioral Score: Add weighted actions (query * 0.5, filter * 0.4, pages * 0.1, clicks * 0.15/0.25), higher for CSNP (0.7/0.6).
- Boosts: Add persona bonuses (e.g., drug clicks ≥ 2: +0.2, CSNP signals: +0.3/0.2).
- Combine: Sum base weight + behavioral score.
- Target Adjust: For target persona, ensure weight ≥ max non-target + 0.15 (0.25 for CSNP).
- Cap: Limit to 1.0 (1.5 for CSNP).
- Normalize: Divide non-CSNP weights by their sum (if > 0).

Example:
- Input: "drug" (target), ma_drug_coverage = 0.8, query_drug = 1, dce_click_count = 2, num_pages_viewed = 3
- Base: min(0.8, 0.5) = 0.5
- Score: (0.5 * 1) + (0.15 * 2) + (0.1 * 3) = 1.1
- Boost: clicks ≥ 2, +0.2 = 1.3
- Combine: 0.5 + 1.3 = 1.8
- Adjust: max non-target = 0.9, max(1.8, 0.9 + 0.15) = 1.8
- Cap: min(1.8, 1.0) = 1.0
- Normalize: if sum = 1.5, w_drug = 1.0 / 1.5 = 0.67
- Result: w_drug = 0.67

Key Notes:
- CSNP gets higher caps and boosts.
- Weights balance plan data and user actions.
- Normalization keeps non-CSNP relative.
