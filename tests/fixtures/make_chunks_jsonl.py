"""Generate a realistic 100-chunk synthetic corpus at tests/fixtures/chunks.jsonl.

This is a natural-language corpus (no embeddings) suitable for exercising the
`ragprobe index` command end to end. Run:

    python tests/fixtures/make_chunks_jsonl.py
"""

from __future__ import annotations

import json
from pathlib import Path

# (source_file, [sentences]) grouped by theme so the corpus has real topology:
# clusters of related facts plus a few cross-cutting "bridge" statements.
THEMES: dict[str, list[str]] = {
    "pricing.md": [
        "The Free plan includes up to 3 projects and community support.",
        "The Pro plan costs $50 per user per month billed monthly.",
        "The Pro plan costs $40 per user per month when billed annually.",
        "The Enterprise plan costs $120 per user per month and includes priority support.",
        "Enterprise customers receive a 20% discount on annual commitments.",
        "Volume discounts begin at 50 seats and scale with seat count.",
        "Nonprofit organizations qualify for a 30% discount with verification.",
        "Add-on storage is billed at $0.10 per GB per month.",
        "The Team plan sits between Pro and Enterprise at $80 per user per month.",
        "Startups in our accelerator program get the Pro plan free for one year.",
    ],
    "billing.md": [
        "Annual billing is invoiced once per year in advance.",
        "Monthly billing is charged on the same calendar day each month.",
        "Invoices are issued in USD by default but EUR and GBP are supported.",
        "Failed payments are retried automatically for up to 7 days.",
        "Receipts are emailed to the account billing contact after each charge.",
        "You can update your payment method from the Billing settings page.",
        "Sales tax and VAT are applied based on your billing address.",
        "Refunds are processed to the original payment method within 10 business days.",
        "Unused prepaid credits roll over for 12 months before expiring.",
        "Purchase orders are accepted for Enterprise contracts over $10,000.",
    ],
    "plan_changes.md": [
        "Tier upgrades require 30 days written notice before the next billing cycle.",
        "Downgrading your plan takes effect at the end of the current billing cycle.",
        "Upgrading takes effect immediately and is prorated for the remainder of the period.",
        "Switching from monthly to annual billing locks in the annual discount.",
        "Canceling a subscription stops renewal but keeps access until period end.",
        "Reactivating a canceled plan restores your previous configuration.",
        "Seat additions are prorated and charged on the next invoice.",
        "Removing seats reduces your charge starting the following billing cycle.",
        "Plan changes are logged in the account audit history.",
        "Annual plans cannot be downgraded mid-term without an early termination fee.",
    ],
    "trials.md": [
        "Free trials last 14 days and do not require a credit card.",
        "The trial unlocks all Pro plan features for evaluation.",
        "Trial data is retained for 30 days after the trial ends.",
        "You can extend a trial once by contacting the sales team.",
        "Converting a trial to a paid plan preserves all your projects.",
        "Trials are limited to one per organization.",
        "A trial countdown banner appears in the dashboard during the trial.",
        "If a trial expires, the workspace becomes read-only until you subscribe.",
        "Trial accounts are rate-limited to 1,000 API calls per day.",
        "Sandbox environments remain available even after a trial ends.",
    ],
    "support.md": [
        "Our support team is available Monday to Friday, 9am to 5pm EST.",
        "Priority support guarantees a first response within 2 business hours.",
        "Community support is provided through the public forum and Discord.",
        "Enterprise customers receive a dedicated customer success manager.",
        "Support tickets can be submitted from the in-app help widget.",
        "Critical outages are tracked publicly on the status page.",
        "Phone support is available only on Enterprise plans.",
        "Documentation and tutorials are free for all users.",
        "Weekend emergency support is available as a paid add-on.",
        "Support response times are measured against your plan's SLA.",
    ],
    "security.md": [
        "All data is encrypted in transit using TLS 1.3.",
        "Data at rest is encrypted with AES-256.",
        "Single sign-on via SAML is available on Enterprise plans.",
        "Two-factor authentication can be enforced for all members.",
        "We are SOC 2 Type II certified and audited annually.",
        "Role-based access control lets admins scope member permissions.",
        "Audit logs capture every administrative action for 1 year.",
        "Customer data is hosted in regional data centers you select.",
        "Penetration tests are conducted by third parties twice a year.",
        "Data deletion requests are honored within 30 days under GDPR.",
    ],
    "api.md": [
        "The REST API is versioned and the current version is v2.",
        "API authentication uses bearer tokens scoped to a workspace.",
        "Rate limits are 60 requests per minute on the Pro plan.",
        "Webhooks deliver events for project, billing, and member changes.",
        "The API returns JSON and uses standard HTTP status codes.",
        "Pagination uses cursor-based tokens for large result sets.",
        "SDKs are available for Python, JavaScript, and Go.",
        "Deprecated endpoints are supported for 6 months after notice.",
        "API keys can be rotated without downtime from the dashboard.",
        "Enterprise plans can request a higher rate limit on demand.",
    ],
    "onboarding.md": [
        "New workspaces start with a guided setup checklist.",
        "You can invite teammates by email or shareable link.",
        "Importing data from CSV is supported during onboarding.",
        "Templates help you create your first project in minutes.",
        "The onboarding wizard configures notifications and integrations.",
        "Admins can preconfigure defaults for all new members.",
        "Sample projects can be loaded to explore features safely.",
        "Onboarding progress is saved and can be resumed anytime.",
        "A product tour highlights key features on first login.",
        "Migration assistance is included for Enterprise customers.",
    ],
    "integrations.md": [
        "Slack integration posts notifications to channels you choose.",
        "The GitHub integration links commits to projects automatically.",
        "Google Drive can be connected to attach files to records.",
        "Zapier connects the product to thousands of other apps.",
        "The Jira integration syncs issues bidirectionally.",
        "Calendar integrations sync deadlines to Google and Outlook.",
        "Integrations are managed from the workspace settings page.",
        "OAuth is used to authorize all third-party integrations.",
        "Some integrations are limited to Team and Enterprise plans.",
        "Integration activity appears in the workspace audit log.",
    ],
    "data.md": [
        "Projects can be exported as JSON or CSV at any time.",
        "Automatic backups run nightly and are retained for 35 days.",
        "You can restore a project to any backup from the last 35 days.",
        "Data residency options include the US, EU, and Asia-Pacific.",
        "Archived projects are excluded from active seat counts.",
        "Bulk operations can update up to 1,000 records at once.",
        "Deleted projects are recoverable from the trash for 30 days.",
        "Custom fields support text, number, date, and select types.",
        "Data import validates rows and reports errors before committing.",
        "Large exports are delivered as a downloadable archive by email.",
    ],
}


def build_chunks() -> list[dict]:
    chunks: list[dict] = []
    counter = 1
    for source, sentences in THEMES.items():
        for page, text in enumerate(sentences, start=1):
            chunks.append(
                {
                    "id": f"chunk_{counter:03d}",
                    "text": text,
                    "metadata": {"source": source, "page": page},
                }
            )
            counter += 1
    return chunks


def main() -> None:
    out = Path(__file__).parent / "chunks.jsonl"
    chunks = build_chunks()
    with out.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    print(f"Wrote {len(chunks)} chunks to {out}")


if __name__ == "__main__":
    main()
