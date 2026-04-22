"""task_16_email_triage variant generator.

Generates varied sets of 13 emails covering the same priority distribution
(1× P0 incident, 1× monitoring alert, 1× high-value client, 2× security/compliance,
2× code review / internal request, 1× newsletter, 1× spam, 4× admin/misc).

The key grading checks (P0 outage, alert linked to outage, client=P1, spam=P4)
are preserved across all variants.
"""
from __future__ import annotations
import random
import textwrap
from typing import Iterator
from . import BaseVariantGenerator, TaskVariant

# ── scenario pools ────────────────────────────────────────────────────────────

_INCIDENTS = [
    {
        "service": "production database cluster",
        "symptom": "Customer-facing services are returning 500 errors.",
        "sender_name": "David Park", "sender_role": "CTO",
        "channel": "#incident-db-20260217",
    },
    {
        "service": "payment processing service",
        "symptom": "All payment transactions are failing with timeout errors.",
        "sender_name": "Rachel Kim", "sender_role": "VP Engineering",
        "channel": "#incident-payments-20260310",
    },
    {
        "service": "authentication service",
        "symptom": "Users cannot log in; SSO tokens are being rejected.",
        "sender_name": "Marcus Webb", "sender_role": "CTO",
        "channel": "#incident-auth-20260401",
    },
    {
        "service": "CDN and static asset delivery",
        "symptom": "Frontend assets are not loading; users see blank pages.",
        "sender_name": "Sofia Moreau", "sender_role": "VP Infrastructure",
        "channel": "#incident-cdn-20260215",
    },
]

_CLIENTS = [
    {"name": "Mike Chen", "company": "BigClient Inc.", "value": "$2M annual contract",
     "ask": "finalize API contract and set up staging credentials"},
    {"name": "Lisa Park", "company": "EnterpriseX Corp.", "value": "$1.5M annual contract",
     "ask": "schedule integration kickoff and share security questionnaire"},
    {"name": "James Torres", "company": "MegaCorp Ltd.", "value": "$3M renewal",
     "ask": "confirm SLA terms and provide SOC 2 report"},
    {"name": "Yuki Tanaka", "company": "GlobalTech SA", "value": "$800K pilot deal",
     "ask": "confirm technical requirements and provide sandbox access"},
]

_SECURITY = [
    {"deadline": "February 19", "action": "rotate passwords and SSH keys"},
    {"deadline": "March 5", "action": "enable MFA on all company systems"},
    {"deadline": "April 12", "action": "complete security awareness training"},
    {"deadline": "January 31", "action": "revoke stale API tokens and rotate secrets"},
]

_CODE_REVIEWS = [
    {"author": "Alice Wong", "pr": "#156", "change": "auth service refactor (OAuth2 PKCE)",
     "blocks": "mobile app release"},
    {"author": "Tom Harris", "pr": "#223", "change": "database migration to Postgres 16",
     "blocks": "Q2 infrastructure upgrade"},
    {"author": "Priya Sharma", "pr": "#89", "change": "async job queue rewrite",
     "blocks": "background processing feature launch"},
    {"author": "Carlos Ruiz", "pr": "#341", "change": "GraphQL schema refactor",
     "blocks": "partner API v2 rollout"},
]

_NEWSLETTERS = [
    "TechDigest Weekly: AI agents are reshaping software development",
    "DevNews: Rust adoption surges in cloud-native development",
    "ByteReport: OpenAI announces next-generation model",
    "CloudWatch Newsletter: Kubernetes 2.0 release date confirmed",
]

_SPAM = [
    ("SaaSTools", "60% off all annual plans - 48 hours only!", "FLASH60"),
    ("DevTools Pro", "Black Friday deal: 70% off premium subscriptions", "BFRIDAY70"),
    ("CloudBoost", "Exclusive offer: free tier upgrade for 3 months", "CLOUDBOOST3"),
    ("InfraPlatform", "Limited offer: enterprise plan at startup prices", "INFRA50"),
]

_ADMIN = [
    ("Benefits enrollment deadline", "Feb 28", "HR portal"),
    ("Performance review self-assessment", "Friday", "HR review form"),
    ("Q1 budget reconciliation", "Thursday", "budget tracker"),
    ("Expense report submission", "end of month", "finance portal"),
    ("Team offsite planning survey", "this Friday", "survey link"),
    ("Annual compliance certification", "March 15", "compliance portal"),
]

_DEPENDABOT_REPOS = [
    ("api-gateway", "#482"),
    ("auth-service", "#91"),
    ("data-pipeline", "#207"),
    ("frontend-app", "#318"),
]

_ALERTS = [
    ("api-gateway", "p99 latency", "3,247ms", "2,000ms"),
    ("payment-service", "error rate", "8.4%", "1%"),
    ("auth-service", "CPU utilization", "97%", "85%"),
    ("search-service", "memory usage", "94%", "80%"),
]


def _make_emails(rng: random.Random) -> dict[str, str]:
    """Generate 13 varied email files for one variant."""
    inc = rng.choice(_INCIDENTS)
    client = rng.choice(_CLIENTS)
    sec = rng.choice(_SECURITY)
    cr = rng.choice(_CODE_REVIEWS)
    nl = rng.choice(_NEWSLETTERS)
    spam_item = rng.choice(_SPAM)
    admin_items = rng.sample(_ADMIN, 4)
    dep_repo, dep_pr = rng.choice(_DEPENDABOT_REPOS)
    alert = rng.choice(_ALERTS)

    emails: dict[str, str] = {}

    # email_01: P0 production incident
    emails["inbox/email_01.txt"] = textwrap.dedent(f"""\
        From: {inc['sender_name'].lower().replace(' ', '.')}@mycompany.com ({inc['sender_name']}, {inc['sender_role']})
        To: me@mycompany.com
        Date: Mon, 17 Feb 2026 08:02:00 -0500
        Subject: URGENT: {inc['service'].title()} outage - all hands needed

        Our {inc['service']} went down at 7:45am EST. {inc['symptom']}
        SRE team is engaged but we need all backend engineers on the war room call immediately.

        War room link: https://meet.mycompany.com/war-room-prod
        Incident channel: {inc['channel']}

        This is a P0 incident. Drop everything else until this is resolved.

        -{inc['sender_name'].split()[0]}
    """)

    # email_02: P2 internal review request
    emails["inbox/email_02.txt"] = textwrap.dedent(f"""\
        From: sarah.marketing@mycompany.com (Sarah Liu, Marketing Director)
        To: me@mycompany.com
        Date: Mon, 17 Feb 2026 09:15:00 -0500
        Subject: Blog post review needed by EOD Wednesday

        Hi,

        We have a new blog post about our Q4 product updates that needs a technical
        accuracy review. It's about 1,200 words. Could you take a look and flag anything
        that's incorrect or misleading? No rush - end of day Wednesday works.

        Draft link: https://docs.mycompany.com/blog-q4-review

        Thanks!
        Sarah
    """)

    # email_03: P3 automated Dependabot PR
    emails["inbox/email_03.txt"] = textwrap.dedent(f"""\
        From: noreply@github.com
        To: me@mycompany.com
        Date: Mon, 17 Feb 2026 07:30:00 -0500
        Subject: [mycompany/{dep_repo}] Pull request {dep_pr}: Dependency updates (Dependabot)

        Dependabot has opened a pull request to update the following dependencies:

        - express: 4.18.2 → 4.19.0 (minor)
        - lodash: 4.17.21 → 4.17.22 (patch)
        - @types/node: 20.10.0 → 20.11.0 (minor)

        All CI checks are passing. No breaking changes detected.

        View pull request: https://github.com/mycompany/{dep_repo}/pull/{dep_pr[1:]}
    """)

    # email_04: P2-P3 administrative (benefits/HR)
    admin4 = admin_items[0]
    emails["inbox/email_04.txt"] = textwrap.dedent(f"""\
        From: jenna.hr@mycompany.com (Jenna Walsh, HR)
        To: all-staff@mycompany.com
        Date: Fri, 14 Feb 2026 16:00:00 -0500
        Subject: Reminder: {admin4[0]} is {admin4[1]}

        Hi everyone,

        Just a friendly reminder that the {admin4[0].lower()} deadline is {admin4[1]}, 2026.
        If you haven't completed this yet, please log into the {admin4[2]} and take action.

        If you have questions, reach out to the HR team.

        Thanks,
        Jenna
    """)

    # email_05: P0-P1 high-value client
    emails["inbox/email_05.txt"] = textwrap.dedent(f"""\
        From: {client['name'].lower().replace(' ', '.')}@{client['company'].lower().replace(' ', '').replace('.', '')}.com ({client['name']}, VP Engineering)
        To: me@mycompany.com
        Date: Mon, 17 Feb 2026 08:45:00 -0500
        Subject: Re: API integration timeline

        Hi,

        Following up on our call last week. Our board approved the integration project
        and we'd like to move forward. We need to {client['ask']} ASAP so our team can start development.

        Can we schedule a 30-minute call this week? Tuesday or Thursday afternoon works best for us.
        This is a {client['value']} so we want to keep momentum.

        Also, our security team will need to complete a vendor assessment. Could you
        send over your SOC 2 report and data processing agreement?

        Best,
        {client['name']}
        VP Engineering, {client['company']}
    """)

    # email_06: P3-P4 LinkedIn social notification
    emails["inbox/email_06.txt"] = textwrap.dedent(f"""\
        From: noreply@linkedin.com
        To: me@mycompany.com
        Date: Sun, 16 Feb 2026 14:22:00 -0500
        Subject: You have 3 new connection requests

        You have new connection requests from:
        - Alex Turner, Software Engineer at TechCorp
        - Maria Santos, Product Manager at StartupXYZ
        - Kevin Park, Recruiter at TopTalent Agency

        View and respond to your invitations:
        https://linkedin.com/notifications
    """)

    # email_07: P2 internal request (perf review / admin)
    admin7 = admin_items[1]
    emails["inbox/email_07.txt"] = textwrap.dedent(f"""\
        From: team-lead@mycompany.com (Rachel Green, Engineering Manager)
        To: me@mycompany.com
        Date: Mon, 17 Feb 2026 09:30:00 -0500
        Subject: {admin7[0]} due {admin7[1]}

        Hi,

        Quick reminder that your {admin7[0].lower()} is due {admin7[1]}.
        Please complete it via the {admin7[2]} I shared last week.

        Let me know if you have any questions.

        Rachel
    """)

    # email_08: P1-P2 security compliance deadline
    emails["inbox/email_08.txt"] = textwrap.dedent(f"""\
        From: security@mycompany.com (Security Team)
        To: engineering@mycompany.com
        Date: Mon, 17 Feb 2026 07:00:00 -0500
        Subject: IMPORTANT: Mandatory {sec['action']} by {sec['deadline']}

        As part of our quarterly security compliance, all engineering team members
        must {sec['action']} by {sec['deadline']}, 2026.

        Required actions:
        1. Complete the required action via https://sso.mycompany.com/reset
        2. Confirm completion by replying to this email

        Failure to comply by the deadline may result in temporary account lockout.

        Security Team
    """)

    # email_09: P3-P4 newsletter
    emails["inbox/email_09.txt"] = textwrap.dedent(f"""\
        From: newsletter@techdigest.io
        To: me@mycompany.com
        Date: Mon, 17 Feb 2026 06:00:00 -0500
        Subject: {nl}

        This week in tech:

        → AI coding agents now write 40% of code at top tech companies
        → New study shows remote engineers are 15% more productive
        → Rust adoption surges in cloud-native development
        → OpenAI announces next-generation model release date
        → Kubernetes 2.0 brings major networking improvements

        Read the full digest: https://techdigest.io/weekly/2026-02-17

        Unsubscribe: https://techdigest.io/unsubscribe
    """)

    # email_10: P2 code review request
    emails["inbox/email_10.txt"] = textwrap.dedent(f"""\
        From: {cr['author'].lower().replace(' ', '.')}@mycompany.com ({cr['author']}, Senior Engineer)
        To: me@mycompany.com
        Date: Mon, 17 Feb 2026 09:50:00 -0500
        Subject: Code review request - {cr['change']}

        Hey,

        I just pushed a {cr['change']} (PR {cr['pr']}). I'd really appreciate your review.

        This change blocks the {cr['blocks']}, so I'd like to merge by Thursday if possible.

        PR link: https://github.com/mycompany/repo/pull/{cr['pr'][1:]}

        Let me know if you need more context.

        Thanks,
        {cr['author']}
    """)

    # email_11: P4 spam/promotional
    spam_sender, spam_subject, spam_code = spam_item
    emails["inbox/email_11.txt"] = textwrap.dedent(f"""\
        From: deals@{spam_sender.lower().replace(' ', '')}.com
        To: me@mycompany.com
        Date: Sat, 15 Feb 2026 10:00:00 -0500
        Subject: 🔥 Flash Sale: {spam_subject}

        LIMITED TIME OFFER!

        Upgrade your development workflow with {spam_sender} Pro.

        Use code {spam_code} at checkout.

        This offer expires Monday at midnight!
    """)

    # email_12: P2 admin/budget/finance
    admin12 = admin_items[2]
    emails["inbox/email_12.txt"] = textwrap.dedent(f"""\
        From: cfo@mycompany.com (Linda Zhao, CFO)
        To: engineering-leads@mycompany.com
        Date: Mon, 17 Feb 2026 08:30:00 -0500
        Subject: {admin12[0]} - action needed by Thursday

        Hi team leads,

        Finance is closing out Q1 budget projections and we need each team to
        complete the {admin12[0].lower()} via the {admin12[2]} by {admin12[1]}.

        Please flag any anticipated overruns or pending purchase requests.

        Thanks,
        Linda
    """)

    # email_13: P0 monitoring alert (correlated with incident)
    svc, metric, current, threshold = alert
    emails["inbox/email_13.txt"] = textwrap.dedent(f"""\
        From: automated-alerts@monitoring.mycompany.com
        To: oncall@mycompany.com, me@mycompany.com
        Date: Mon, 17 Feb 2026 07:48:00 -0500
        Subject: [ALERT] {svc} {metric} exceeding threshold - {metric} > {threshold}

        MONITORING ALERT

        Service: {svc}
        Metric: {metric}
        Current value: {current} (threshold: {threshold})
        Duration: 3 minutes
        Status: FIRING

        Dashboard: https://grafana.mycompany.com/d/api-latency
        Runbook: https://wiki.mycompany.com/runbooks/{svc}

        This alert correlates with the ongoing {inc['service']} incident (INC-20260217-001).
    """)

    return emails


class Generator(BaseVariantGenerator):
    task_id = "task_16_email_triage"

    def sample(self, n: int = 1, seed: int | None = None) -> Iterator[TaskVariant]:
        rng = random.Random(seed)
        for i in range(n):
            variant_rng = random.Random(rng.randint(0, 2**31))
            workspace_files = _make_emails(variant_rng)
            yield TaskVariant(
                task_id=self.task_id,
                prompt=(
                    "You are helping triage an overflowing email inbox. The emails have been provided "
                    "to you in the `inbox/` folder in your workspace (files named `email_01.txt` through "
                    "`email_13.txt`). Read all 13 emails and create a triage report saved to "
                    "`triage_report.md`. For each email, assign:\n\n"
                    "1. **Priority**: P0 (drop everything), P1 (today), P2 (this week), "
                    "P3 (when convenient), P4 (no action / archive)\n"
                    "2. **Category**: one of \"incident\", \"client\", \"internal-request\", "
                    "\"administrative\", \"code-review\", \"automated\", \"newsletter\", \"spam\"\n"
                    "3. **Recommended action**: a brief (1-2 sentence) description of what to do\n\n"
                    "Organize the report with emails sorted by priority (most urgent first). "
                    "Include a brief summary section at the top that highlights the most critical "
                    "items and suggests a plan for the day."
                ),
                workspace_files=workspace_files,
                metadata={"num_emails": 13},
            )
