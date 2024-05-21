import json
import os
from copy import deepcopy


def create_nb(source, template, output_path: str):
    snippets = {}
    snip = []
    for line in source:
        if line.startswith("# !!!"):
            name = line[5:].strip()
            if name in snippets:
                print(f"[!] Snippet {name} defined more than once")
            snip = []
            snippets[name] = snip
        else:
            snip.append(line)
    for snippet in snippets.values():
        while len(snippet) > 0 and len(snippet[-1].strip()) == 0:
            snippet.pop()

    used_snippets = set()
    output = deepcopy(template)

    for cell in output["cells"]:
        if cell["cell_type"] == "code":
            template_source = cell["source"]
            cell["source"] = []

            for line in template_source:
                if line.startswith("# !!!"):
                    snippet_name = line[5:].strip()
                    if snippet_name not in snippets:
                        print(f"[!] Snippet {snippet_name} requested but not provided")
                    else:
                        if snippet_name in used_snippets:
                            print(f"[!] Snippet {snippet_name} used more than once")
                        else:
                            print(f"Used snippet {snippet_name}")
                        
                        cell["source"] += snippets[snippet_name]
                        used_snippets.add(snippet_name)
                else:
                    cell["source"].append(line)
    
    for snippet_name in snippets:
        if snippet_name not in used_snippets:
            print(f"[!] Snippet {snippet_name} provided but not requested")
    
    with open(output_path, "w") as output_file:
        json.dump(output, output_file)


cwd = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(cwd)
templates = [f for f in files if f.startswith("template")]
instatiators = [f for f in files if f not in templates and f.endswith(".ipynb")]

print(f"Generating notebooks based on {instatiators}")
print(f"using the templates {templates}")

template_src = {}
for nb_name in templates:
    with open(os.path.join(cwd, nb_name)) as nb:
        template_src[nb_name] = json.load(nb)

for nb_name in instatiators:
    with open(os.path.join(cwd, nb_name)) as nb:
        cells = json.load(nb)["cells"]
        for cell in cells:
            if cell["cell_type"] == "code":
                source = cell["source"]
                if source[0].startswith("# ***"):
                    template, _, output = source[0][6:].partition("->")
                    output = output.strip()
                    outdir = os.path.join(os.path.dirname(cwd), "generated", nb_name.partition(".")[0])
                    os.makedirs(outdir, exist_ok=True)
                    output = os.path.join(outdir, output)

                    print(f"> Creating notebook: {nb_name} {template.strip()} -> {output.strip()}")
                    template = template_src[template.strip()]
                    create_nb(source, template, output)
