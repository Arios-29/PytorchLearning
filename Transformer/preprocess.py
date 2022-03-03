def read_data_nmt():
    """载入数据文件"""
    with open("data/fra.txt", 'r', encoding='utf-8') as file:
        return file.read()


def preprocess_nmt(text):
    """将大写字母改为小写, 在单词和标点符号之间插入空格"""

    def is_not_space(char, pre_char):
        return char in set(',.!?') and pre_char != " "

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and is_not_space(char, text[i - 1]) else char for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """
    :param text:
    :param num_examples:
    :return: 返回num_examples个样本, source[i]表示第i个源语言序列, target[i]为对应的目标语言序列
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


if __name__ == "__main__":
    text = read_data_nmt()
    print(text.__class__)
