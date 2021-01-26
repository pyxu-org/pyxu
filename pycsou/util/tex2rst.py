import re
from typing import Optional


def beamer2rst(input_file, output_file: Optional[str] = None):
    if output_file is None:
        output_file = f'{input_file.split(".")[0]}.rst'
    with open(output_file, 'w') as out_file:
        with open(input_file, 'r') as in_file:
            file_content = in_file.read()
        frames = re.findall(r'\\begin\{frame\}(.*?)\\end\{frame\}', file_content, flags=re.DOTALL)
        for frame in frames:
            frame = re.sub(r'\$\$(?P<EQ>.*?)\$\$', r'\n.. math::\n\n   \g<EQ>', frame, flags=re.DOTALL)
            frame = re.sub(r'\$(?P<EQ>.*?)\$', r':math:`\g<EQ>`', frame, flags=re.DOTALL)
            frame = re.sub(r'\\begin\{equation\}(?P<EQ>.*?)\\end\{equation\}', r'\n.. math::\n\n   \g<EQ>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{equation\*\}(?P<EQ>.*?)\\end\{equation\*\}', r'\n.. math::\n\n   \g<EQ>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{align\*\}(?P<EQ>.*?)\\end\{align\*\}', r'\n.. math::\n\n   \g<EQ>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{align\}(?P<EQ>.*?)\\end\{align\}', r'\n.. math::\n\n   \g<EQ>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{itemize\}(?P<items>.*?)\\end\{itemize\}', r'\n\g<items>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\begin\{enumerate\}(?P<items>.*?)\\end\{enumerate\}', r'\n\g<items>\n', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\item\s*(?P<item>\S.*?)', r'* \g<item>', frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\cite\[(?P<details>.*?)\]\{(?P<citation>.*?)\}', r'\g<details> of [\g<citation>]_',
                           frame,
                           flags=re.DOTALL)
            frame = re.sub(r'\\(?P<symbol>[a-zA-Z])cal', r'\\mathcal{\g<symbol>}', frame)
            frame = re.sub(r'\\(?P<symbol>[a-zA-Z])scr', r'\\mathcal{\g<symbol>}', frame)
            frame = re.sub(r'\\(?P<symbol>[a-zA-Z])bf', r'\\mathbf{\g<symbol>}', frame)
            frame = re.sub(r'\\(?P<symbol>[a-zA-Z])bb', r'\\mathbb{\g<symbol>}', frame)
            frame = re.sub(r'\\bb(?P<symbol>[a-zA-Z])', r'\\mathbf{\g<symbol>}', frame)
            frame = re.sub(r'\\R', r'\\mathbb{R}', frame)
            frame = re.sub(r'\\N', r'\\mathbb{N}', frame)
            frame = re.sub(r'\\Q', r'\\mathbb{Q}', frame)
            frame = re.sub(r'\\bm\{(?P<symbol>.*?)\}', r'\\mathbf{\g<symbol>}', frame)
            frame = re.sub(r'\\emph\{(?P<expression>.*?)\}', r'*\g<expression>*', frame)
            frame = re.sub(r'\\textbf\{(?P<expression>.*?)\}', r'**\g<expression>**', frame)
            frame = re.sub(r'(\\green|\\blue|\\red|\\orange|\\purple)\{(?P<expression>.*?)\}', r'\g<expression>',
                           frame)
            out_file.write(frame)


if __name__ == '__main__':
    beamer2rst('pycsou/util/part2.tex')
