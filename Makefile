PYTHON := python

all: kanji-old-new-form.json ivs-svs-base-mappings.json

download-glyphwiki-dump: glyphwiki-dump.tar.gz

glyphwiki-dump.tar.gz:
	curl -L -o glyphwiki-dump.tar.gz https://glyphwiki.org/dump.tar.gz

expand-glyphwiki-dump: glyphwiki-dump.tar.gz
	install -d glyphwiki-dump
	tar -C glyphwiki-dump -xvf $<

glyphwiki.sqlite3: glyphwiki-dump/dump_newest_only.txt
	$(PYTHON) gen_tables.py gen-glyphwiki-sqlite3 -i $< -o $@

radicals.template.json: glyphwiki.sqlite3
	$(PYTHON) gen_tables.py gen-table-cjk-radicals --glyphwiki-db=$< -o $@

aj17-kanji.json: Adobe-Japan1/aj17-kanji.txt 
	$(PYTHON) gen_tables.py gen-table-ivd-adobe-japan1 -i $< -o $@

kanji-old-new-form.json: cjkvi-variants/jp-old-style.txt aj17-kanji.json 
	$(PYTHON) gen_tables.py gen-table-old-new --ivd-table=aj17-kanji.json -i $< -o $@

ivs-svs-base-mappings.json: aj17-kanji.json 
	$(PYTHON) gen_tables.py gen-table-ivs-svs-basic -i $< -o $@

adobe-japan1-hanyo-denshi.json: glyphwiki.sqlite3 aj17-kanji.json
	$(PYTHON) gen_tables.py gen-table-ivd-mapping-adobe-japan1-and-hanyo-denshi --glyphwiki-db=glyphwiki.sqlite3 --ivd-table=aj17-kanji.json -o $@


.PHONY: all download-glyphwiki-dump expand-glyphwiki-dump
