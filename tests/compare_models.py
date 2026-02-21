import ollama
import time
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()

MODELS = ["phi3:mini", "mistral:7b", "qwen2.5-coder:7b"]

TESTS = [
    ("Chat / Q&A",      "Explain black holes to a 10 year old in 5 sentences."),
    ("Reasoning",       "If a train travels 60km/h for 2.5 hours, then 80km/h for 1.5 hours, what is the total distance?"),
    ("Code Gen",        "Write a Python function to find all prime numbers up to n using the Sieve of Eratosthenes."),
    ("Debug Code",      "Find and fix the bug:\ndef avg(nums):\n  return sum(nums) / len(nums)\nprint(avg([]))"),
    ("Summarization",   "Summarize in 2 sentences: Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information using connectionist approaches. Deep learning uses multiple layers to progressively extract higher-level features from raw input."),
    ("Creativity",      "Write a 4-line poem about artificial intelligence."),
]

results = {model: [] for model in MODELS}

for category, prompt in TESTS:
    console.rule(f"[bold yellow]📝 {category}")
    console.print(f"[dim]Prompt: {prompt}[/dim]\n")

    for model in MODELS:
        console.print(f"[cyan]🤖 Running: {model}[/cyan]")
        try:
            start = time.time()
            response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            elapsed = time.time() - start

            answer = response['message']['content']
            tokens = response.get('eval_count', 0)
            tok_per_sec = round(tokens / elapsed, 2) if elapsed > 0 else 0

            console.print(f"[green]Response:[/green] {answer}\n")
            console.print(f"[dim]⏱ {elapsed:.2f}s | 🔢 {tokens} tokens | ⚡ {tok_per_sec} tok/s[/dim]\n")

            results[model].append({
                "category": category,
                "time": round(elapsed, 2),
                "tokens": tokens,
                "tok_per_sec": tok_per_sec
            })

        except Exception as e:
            console.print(f"[red]❌ Error with {model}: {e}[/red]\n")
            results[model].append({
                "category": category,
                "time": 0, "tokens": 0, "tok_per_sec": 0
            })

# ── Summary Table ──────────────────────────────────────
console.rule("[bold magenta]📊 FINAL COMPARISON SUMMARY")

table = Table(show_lines=True, title="Model Benchmark — phi3:mini vs mistral:7b vs qwen2.5-coder:7b")
table.add_column("Category",             style="cyan",   min_width=15)
table.add_column("phi3:mini\nTok/s",     style="green",  justify="right")
table.add_column("mistral:7b\nTok/s",    style="yellow", justify="right")
table.add_column("qwen2.5-coder\nTok/s", style="magenta",justify="right")
table.add_column("🏆 Fastest",           style="bold white")

for i, (category, _) in enumerate(TESTS):
    row = []
    speeds = []
    for model in MODELS:
        r = results[model][i]
        row.append(f"{r['tok_per_sec']}")
        speeds.append((r['tok_per_sec'], model))
    fastest = max(speeds, key=lambda x: x[0])[1].split(":")[0]
    table.add_row(category, *row, fastest)

console.print(table)

# ── Average Speed per Model ────────────────────────────
console.print("\n[bold]Average Speed Summary:[/bold]")
for model in MODELS:
    avg = round(sum(r['tok_per_sec'] for r in results[model]) / len(results[model]), 2)
    total_tokens = sum(r['tokens'] for r in results[model])
    total_time = round(sum(r['time'] for r in results[model]), 2)
    console.print(f"  [cyan]{model:<25}[/cyan] avg [green]{avg}[/green] tok/s | total [yellow]{total_tokens}[/yellow] tokens in [dim]{total_time}s[/dim]")

console.print("\n[bold green]✅ Comparison complete![/bold green]")