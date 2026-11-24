"""
Rich CLI for Resume Scorer & Gap Analyzer (BERT-Powered).
Run:  python cli.py <resume_path> <jd_path>
"""

import sys
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich import box
import typer

from analyzer.scorer import ResumeScorer

console = Console()
app = typer.Typer(
    name="resume-scorer",
    help="🎯 BERT-Powered Resume Scorer & Gap Analysis CLI",
    add_completion=False,
)


def score_color(score: float) -> str:
    if score >= 80:
        return "bold green"
    elif score >= 65:
        return "bold yellow"
    elif score >= 50:
        return "bold dark_orange"
    else:
        return "bold red"


def status_icon(status: str) -> str:
    return {"strong_match": "✅", "moderate_match": "🟡", "gap": "❌"}.get(status, "❓")


def save_to_supabase(resume_text: str, jd_text: str, result, resume_source: str, jd_source: str):
    """Attempt to save analysis results to Supabase. Silently skips if not configured."""
    try:
        from analyzer.database import ResumeDB

        db = ResumeDB()

        # Determine file name if applicable
        resume_path = Path(resume_source)
        file_name = resume_path.name if resume_path.is_file() else None

        # Save resume
        resume_row = db.save_resume(
            raw_text=resume_text,
            file_name=file_name,
        )

        # Save JD
        jd_row = db.save_job_description(raw_text=jd_text)

        # Save analysis
        if resume_row.get("id") and jd_row.get("id"):
            analysis_row = db.save_analysis(
                resume_id=resume_row["id"],
                jd_id=jd_row["id"],
                result_obj=result,
            )
            console.print(
                f"\n  [green]✅ Saved to Supabase[/green] — Analysis ID: [cyan]{analysis_row.get('id', 'N/A')}[/cyan]"
            )
            return True
    except ConnectionError:
        # .env not configured — silently skip
        console.print("\n  [dim]ℹ️  Supabase not configured. Results not saved to database. Set up .env to enable.[/dim]")
    except Exception as e:
        console.print(f"\n  [yellow]⚠️  Could not save to Supabase: {e}[/yellow]")
    return False


@app.command()
def analyze(
    resume: str = typer.Argument(..., help="Path to resume file (PDF or TXT) or raw text"),
    jd: str = typer.Argument(..., help="Path to job description file (TXT) or raw text"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Print results as JSON"),
    no_save: bool = typer.Option(False, "--no-save", help="Skip saving to Supabase"),
):
    """
    Analyze a resume against a job description and produce a detailed gap report.
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🎯 BERT-Powered Resume Scorer & Gap Analyzer[/bold cyan]\n"
        "[dim]Semantic similarity • Skill taxonomy • ATS alignment[/dim]",
        border_style="bright_cyan",
    ))
    console.print()

    # Validate inputs
    resume_path = Path(resume)
    jd_path = Path(jd)

    if resume_path.is_file():
        console.print(f"  📄 Resume : [cyan]{resume_path}[/cyan]")
    else:
        console.print(f"  📄 Resume : [dim](raw text, {len(resume)} chars)[/dim]")

    if jd_path.is_file():
        console.print(f"  📋 JD     : [cyan]{jd_path}[/cyan]")
    else:
        console.print(f"  📋 JD     : [dim](raw text, {len(jd)} chars)[/dim]")

    console.print()

    with console.status("[bold green]Loading BERT model and computing embeddings…[/bold green]"):
        scorer = ResumeScorer()
        result = scorer.analyze(resume, jd)

    # Save to Supabase
    if not no_save:
        resume_text = scorer.resume_parser.extract_text(resume)
        jd_text = scorer.resume_parser.extract_text(jd)
        save_to_supabase(resume_text, jd_text, result, resume, jd)

    if json_output:
        console.print_json(result.model_dump_json(indent=2))
        return

    # ── Overall Score ──────────────────────────────────────────────
    color = score_color(result.overall_score)
    console.print(Panel(
        f"[{color}]  ★  OVERALL SCORE: {result.overall_score}%  ★[/{color}]\n\n"
        f"  [bold]{result.match_tier}[/bold]\n\n"
        f"  {result.summary}",
        title="[bold white]📊 Analysis Result[/bold white]",
        border_style="bright_cyan",
        padding=(1, 3),
    ))

    # ── Sub-Scores Table ──────────────────────────────────────────
    scores_table = Table(title="Sub-Dimension Scores", box=box.ROUNDED, show_lines=True)
    scores_table.add_column("Dimension", style="bold", min_width=22)
    scores_table.add_column("Score", justify="center", min_width=10)
    scores_table.add_column("Bar", min_width=25)

    for label, val in [
        ("Semantic Alignment", result.semantic_score),
        ("Skills Match", result.skills_score),
        ("Experience", result.experience_score),
        ("Education", result.education_score),
    ]:
        bar_len = int(val / 100 * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        sc = score_color(val)
        scores_table.add_row(label, f"[{sc}]{val}%[/{sc}]", f"[{sc}]{bar}[/{sc}]")

    console.print(scores_table)
    console.print()

    # ── Skills Analysis ───────────────────────────────────────────
    skills_table = Table(title="🔑 Skills Analysis", box=box.ROUNDED, show_lines=True)
    skills_table.add_column("Status", justify="center", min_width=10)
    skills_table.add_column("Skills", min_width=50)

    if result.matched_skills:
        skills_table.add_row(
            "[green]✅ Matched[/green]",
            ", ".join(f"[green]{s}[/green]" for s in result.matched_skills)
        )
    if result.missing_skills:
        skills_table.add_row(
            "[red]❌ Missing[/red]",
            ", ".join(f"[red]{s}[/red]" for s in result.missing_skills)
        )
    if result.extra_skills:
        skills_table.add_row(
            "[blue]➕ Extra[/blue]",
            ", ".join(f"[blue]{s}[/blue]" for s in result.extra_skills)
        )

    console.print(skills_table)
    console.print()

    # ── Category Breakdown ────────────────────────────────────────
    if result.skills_by_category:
        cat_table = Table(title="📂 Skills by Category", box=box.SIMPLE_HEAVY)
        cat_table.add_column("Category", style="bold", min_width=28)
        cat_table.add_column("JD", justify="center")
        cat_table.add_column("Matched", justify="center")
        cat_table.add_column("Missing", justify="center")
        cat_table.add_column("Missing Skills")

        for cat, stats in result.skills_by_category.items():
            display_cat = cat.replace("_", " ").title()
            missing_str = ", ".join(stats.missing_skills) if stats.missing_skills else "—"
            cat_table.add_row(
                display_cat,
                str(stats.total_jd),
                f"[green]{stats.matched}[/green]",
                f"[red]{stats.missing}[/red]" if stats.missing > 0 else "[green]0[/green]",
                missing_str,
            )

        console.print(cat_table)
        console.print()

    # ── Requirements Semantic Alignment ───────────────────────────
    if result.requirement_matches:
        req_table = Table(
            title=f"🎯 JD Requirements Alignment ({result.strong_matches_count} strong | "
                  f"{result.moderate_matches_count} moderate | {result.gaps_count} gaps)",
            box=box.ROUNDED,
            show_lines=True,
        )
        req_table.add_column("", justify="center", width=3)
        req_table.add_column("Requirement", min_width=45, max_width=55)
        req_table.add_column("Score", justify="center", min_width=8)
        req_table.add_column("Best Match in Resume", min_width=40, max_width=50)

        for m in result.requirement_matches:
            icon = status_icon(m.status)
            sc = score_color(m.similarity_score)
            match_text = (m.best_matching_text or "")[:80]
            req_text = m.requirement[:80]
            req_table.add_row(icon, req_text, f"[{sc}]{m.similarity_score}%[/{sc}]", match_text)

        console.print(req_table)
        console.print()

    # ── Experience & Education ────────────────────────────────────
    exp_edu_table = Table(title="🏢 Experience & 🎓 Education", box=box.ROUNDED, show_lines=True)
    exp_edu_table.add_column("Dimension", style="bold", min_width=15)
    exp_edu_table.add_column("Status", justify="center", min_width=15)
    exp_edu_table.add_column("Details", min_width=50)

    exp_status_style = {
        "meets": "[green]✅ Meets[/green]",
        "exceeds": "[green]⭐ Exceeds[/green]",
        "underqualified": "[red]⚠️ Gap[/red]",
        "undetermined": "[dim]— N/A[/dim]",
    }

    console.print()
    exp_edu_table.add_row(
        "Experience",
        exp_status_style.get(result.experience.status, result.experience.status),
        result.experience.details,
    )
    exp_edu_table.add_row(
        "Education",
        exp_status_style.get(result.education.status, result.education.status),
        result.education.details,
    )
    console.print(exp_edu_table)
    console.print()

    # ── Section Relevance ─────────────────────────────────────────
    if result.sections:
        sec_table = Table(title="📑 Resume Section Relevance", box=box.SIMPLE)
        sec_table.add_column("Section", style="bold", min_width=15)
        sec_table.add_column("Present", justify="center")
        sec_table.add_column("Words", justify="center")
        sec_table.add_column("JD Similarity", justify="center")

        for sec in result.sections:
            present_icon = "[green]✅[/green]" if sec.present else "[red]❌[/red]"
            sc = score_color(sec.similarity_to_jd)
            sec_table.add_row(
                sec.name,
                present_icon,
                str(sec.word_count),
                f"[{sc}]{sec.similarity_to_jd}%[/{sc}]" if sec.present else "[dim]—[/dim]",
            )

        console.print(sec_table)
        console.print()

    # ── Recommendations ───────────────────────────────────────────
    if result.recommendations:
        rec_panel = "\n".join(f"  {r}" for r in result.recommendations)
        console.print(Panel(
            rec_panel,
            title="[bold yellow]💡 Actionable Recommendations[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
        console.print()

    # ── Bullet Improvement Suggestions ────────────────────────────
    if result.bullet_improvement_suggestions:
        console.print("[bold magenta]✏️  Bullet Point Enhancement Tips:[/bold magenta]")
        for i, sug in enumerate(result.bullet_improvement_suggestions, 1):
            console.print(f"  {i}. [dim]Original:[/dim] {sug.get('original', '')[:80]}")
            console.print(f"     [yellow]Tip:[/yellow] {sug.get('tip', '')}")
            console.print()

    # ── ATS Formatting Tips ───────────────────────────────────────
    if result.ats_formatting_tips:
        tips_text = "\n".join(f"  • {t}" for t in result.ats_formatting_tips)
        console.print(Panel(
            tips_text,
            title="[bold blue]📐 ATS Formatting Tips[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        ))
        console.print()


@app.command("history")
def show_history():
    """Show past analysis results from Supabase."""
    try:
        from analyzer.database import ResumeDB
        db = ResumeDB()
    except ConnectionError as e:
        console.print(f"[red]{e}[/red]")
        return

    analyses = db.list_analyses(limit=15)
    if not analyses:
        console.print("[dim]No past analyses found.[/dim]")
        return

    table = Table(title="📜 Analysis History", box=box.ROUNDED, show_lines=True)
    table.add_column("ID", max_width=8)
    table.add_column("Score", justify="center")
    table.add_column("Tier")
    table.add_column("Resume")
    table.add_column("Job")
    table.add_column("Date")

    for a in analyses:
        sc = score_color(a.get("overall_score", 0))
        resume_info = a.get("resumes", {})
        jd_info = a.get("job_descriptions", {})
        table.add_row(
            str(a.get("id", ""))[:8],
            f"[{sc}]{a.get('overall_score', 0)}%[/{sc}]",
            a.get("match_tier", ""),
            resume_info.get("candidate_name") or resume_info.get("file_name") or "—",
            jd_info.get("title") or jd_info.get("company") or "—",
            str(a.get("created_at", ""))[:16],
        )

    console.print(table)


if __name__ == "__main__":
    app()
