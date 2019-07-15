# attention-is-all-you-need-pytorch code analysis
*Junha Hyung*

## preprocess.py
### main()
```python
opt.max_word_seq_len
```

### read_instances_from_file(inst_file, max_sent_len, keep_case)
- change into lower case if not have `keep_case` option
- 문장 길이가 max_sent_len 보다 클 경우 max로 자른다 (word_inst)
- word_inst 가 존재하면 앞뒤에 \<s> <\s> 를 붙여 [['\<s>', 'a', 'b', 'c', '<\s>']]꼴로 word_insts에 append
- 존재하지 않는다면 [None] 을 append
- `return word_insts`

>Constants.BOS_WORD: '\<s>'
Constants.EOS_WORD: '\</s>'
Constants.PAD_WORD: '\<blank>'
Constants.UNK_WORD: '\<unk>'

## build_vocab_idx(word_insts, min_word_count)
- `full vocab`: 모든 단어를 모아둔 set
- `word_count`: 단어들의 출현 횟수를 저장하는 dictionary
- `word2idx`: 단어들을 0~len-1까지 인덱싱, 이때 `min_word_count`를 넘지 못하면 무시
- `return word2idx`

## convert_instance_to_idx_seq(word_insts, word2idx):
- replace all words in word_insts into an index (unk for words not in word2idx)
