import ollama
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()
MODEL = "phi3:mini"

def run_test(label, prompt, category):
    console.rule(f"[bold cyan]🧪 {category} — {label}")
    console.print(f"[yellow]Prompt:[/yellow] {prompt}\n")
    
    start = time.time()
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    elapsed = time.time() - start
    
    answer = response['message']['content']
    tokens = response.get('eval_count', 0)
    tok_per_sec = round(tokens / elapsed, 2) if elapsed > 0 else 0
    
    console.print(Panel(answer, title="Response", border_style="green"))
    console.print(f"[dim]⏱ {elapsed:.2f}s | 🔢 {tokens} tokens | ⚡ {tok_per_sec} tok/s[/dim]\n")
    
    return {"category": category, "label": label, "time": round(elapsed, 2), "tokens": tokens, "tok_per_sec": tok_per_sec}

tests = [
    ("Simple Fact",    "What is the capital of Australia and what is it known for?",                                       "Chat / Q&A"),
    ("Reasoning",      "You have a 3L and 5L jug. How do you measure exactly 4L?",                                         "Chat / Q&A"),
    ("Write Function", "Write a Python palindrome checker with docstring and examples.",                                    "Code Gen"),
    ("Debug Code",     "Fix and optimize this with memoization:\ndef fib(n):\n  return fib(n-1)+fib(n-2)",                 "Code Gen"),
    ("Summarize",      "Summarize in 3 bullets: LLMs are trained on vast text, can generate human-like output, but hallucinate and carry bias.", "Summarization"),
    ("Speed: Short",   "Name 5 European countries.",                                                                       "Benchmark"),
    ("Speed: Long",    "Write a 200-word essay on why history matters.",                                                   "Benchmark"),
]

results = []
for label, prompt, category in tests:
    results.append(run_test(label, prompt, category))

# Summary Table
table = Table(title="📊 Benchmark Summary", show_lines=True)
table.add_column("Category",  style="cyan")
table.add_column("Test",      style="white")
table.add_column("Time",      style="yellow", justify="right")
table.add_column("Tokens",    style="blue",   justify="right")
table.add_column("Tok/s",     style="green",  justify="right")

for r in results:
    table.add_row(r['category'], r['label'], f"{r['time']}s", str(r['tokens']), str(r['tok_per_sec']))

console.print(table)
avg = round(sum(r['tok_per_sec'] for r in results) / len(results), 2)
console.print(f"\n[bold green]🏁 Average Speed: {avg} tok/s on {MODEL}[/bold green]")