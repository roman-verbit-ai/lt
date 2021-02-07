import re
from pathlib import Path

# regex list
re_p_content = re.compile(r"<P>.*?</P>", re.DOTALL)
re_a_content = re.compile(r"<A.*?</A>", re.DOTALL)
re_b_content = re.compile(r"<B>.*?</B>", re.DOTALL)
re_tag = re.compile(r"</?.*?>", re.DOTALL)
re_empty_lines = re.compile(r"\n\s*\n")
re_parenth = re.compile(r"[{(].*?[})]")


def extract_file_text(file_path, encoding="Windows-1255"):

    # read file content
    file_content = Path(file_path).read_text(encoding=encoding)

    # find matches
    matches = re.finditer(re_p_content, file_content)

    # iterate <P>s
    text_lines = list()
    for matchNum, match in enumerate(matches, start=1):

        # get body
        p_body = match.group()

        # remove html tags
        p_body = re.sub(re_a_content, '', p_body, 0)
        p_body = re.sub(re_b_content, '', p_body, 0)
        p_body = re.sub(re_tag, '', p_body, 0)

        # clean parentheticals
        p_body = re.sub(re_parenth, '', p_body, 0)

        # remove empty lines and spaces
        p_body = p_body.replace(u'\xa0', ' ')
        p_body = re.sub(re_empty_lines, '\n', p_body, 0)

        # iterate lines
        for line in p_body.split('\n'):
            text_lines.append(line.strip())

    return text_lines


def gen_clean_corpus(corpus_path):

    # corpus
    corpus = list()

    # iterate files
    for file_path in Path(corpus_path).glob('*.htm'):

        # read file
        clean_file_lines = extract_file_text(file_path)

        # add to corpus
        corpus.extend(clean_file_lines)

    # dump corpus
    corpus_string = '\n'.join(corpus)
    Path(corpus_path).with_name(f'{Path(corpus_path).stem}.txt').write_text(corpus_string, encoding="utf-8")


# gen corpus
gen_clean_corpus("/Users/himmelroman/projects/lt/data/heb_bible_text_only")
