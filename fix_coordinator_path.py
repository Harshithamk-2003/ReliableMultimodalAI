path = "src/agent/coordinator_agent.py"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

old = 'sys.path.append("D:\\\\ReliableMultimodalAI\\\\src\\\\fusion")'
new = (
    "import os\n"
    "_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n"
    'sys.path.append(os.path.join(_BASE_DIR, "src", "fusion"))'
)

if old not in content:
    print("WARNING: exact old string not found. No changes made.")
else:
    content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed coordinator_agent.py successfully.")