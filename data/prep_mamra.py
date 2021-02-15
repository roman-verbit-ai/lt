import re
from pathlib import Path
from encodings.aliases import aliases

# regex list
re_p_content = re.compile(r"<P>.*?</P>", re.DOTALL)
re_a_content = re.compile(r"<A.*?</A>", re.DOTALL)
re_b_content = re.compile(r"<B>.*?</B>", re.DOTALL)
re_tag = re.compile(r"</?.*?>", re.DOTALL)
re_empty_lines = re.compile(r"\n\s*\n")
re_parenth = re.compile(r"[{(].*?[})]")

# encodings
heb_encodings = ['utf-8', 'cp1255', 'iso8859_8', 'cp424', 'cp856', 'cp862']
other_encodings = set(aliases.values()) - set(heb_encodings)


def read_heb_file(file_path):

    # try hebrew encodings
    file_content = read_file(file_path, heb_encodings)
    if not file_content:

        # try all other encodings
        file_content = read_file(file_path, other_encodings)

    return file_content


def read_file(file_path, enc_list):

    # try all encodings
    for enc in enc_list:

        try:

            # read file content
            file_content = Path(file_path).read_text(encoding=enc)

            # verify hebrew
            if 'א' in file_content or 'ב' in file_content:
                return file_content

        except:
            pass


def extract_file_text(file_path):

    # read file content
    file_content = read_heb_file(file_path)
    if not file_content:
        print(f'Failed to process file: {file_path}')
        return

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
        p_body = p_body.replace(u'\xa0', ' ').replace('\u200d', '')
        p_body = re.sub(re_empty_lines, '\n', p_body, 0)

        # iterate lines
        for line in p_body.split('\n'):
            text_lines.append(line.strip())

    return text_lines


def gen_clean_corpus(corpus_path):

    # corpus
    corpus = list()
    count = 0

    # iterate files
    file_list = list(sorted(Path(corpus_path).glob('*.htm')))
    for file_path in file_list:

        # read file
        clean_file_lines = extract_file_text(file_path)
        if clean_file_lines:

            # add to corpus
            count += 1
            corpus.extend(clean_file_lines)

    # dump corpus
    print(f'Total: {len(file_list)}, Success: {count}')
    corpus_string = '\n'.join(corpus)
    Path(corpus_path).with_name(f'{Path(corpus_path).stem}.txt').write_text(corpus_string, encoding="utf-8")


# gen corpus
gen_clean_corpus("/Users/himmelroman/projects/lt/data/corpora/heb_bible_accents")

# load corpus
text = Path("/Users/himmelroman/projects/lt/data/corpora/heb_bible_accents.txt").read_text()
print(set(text))
