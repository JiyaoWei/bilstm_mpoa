import json
# import fire
import jieba
class GenHtml(object):
    def __init__(self):
        self.json_path = 'result/attn_data.json'

    def load_json(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)
        return self.json_data

    def highlight(self, word, attn):
        html_color = '#%02X%02X%02X' % (255, int(255 * (1 - attn)), int(255 * (1 - attn)))
        return '<span style="background-color: {}">{}</span>'.format(html_color, word)

    def mk_html(self, seqs_token, attns):
        html = ""
        if attns == 1:
            attns = [1]
        for ix, attn in zip(seqs_token, attns):
            html += ' ' + self.highlight(ix, attn)
        return '<li>' + html + '</li>'

    def mk_label(self,label,pre_label):
        html = ''
        html = html + '<span>' + "真实标签：" + str(label) + '</span>' + " | "
        html = html + '<span>' + "预测标签：" + str(pre_label) + '</span>'
        return '<li>' + html + '</li>'

    def gen_html(self,args):
        seqs = self.json_data['sequences']
        attns = self.json_data['attention_weights']
        label = self.json_data['rea_labels']
        pre_label = self.json_data['pre_labels']
        batch_size = len(label)
        text = '<ul>'
        for i in range(batch_size):
            text += self.mk_label(label[i],pre_label[i])
            if args.attention_layer == 'mpoa' or  args.attention_layer == 'mpa':
                for j in range(args.num_classes):
                    seq_tokens = jieba.lcut(seqs[i][0])
                    text += self.mk_html(seq_tokens, attns[i*3+j])
            else:
                seq_tokens = jieba.lcut(seqs[i][0])
                text += self.mk_html(seq_tokens, attns[i])
        text += '</ul>'
        self.text = text

    def save(self,save_path):
        with open(save_path, 'w', encoding='utf-8') as f:
            head = '''
            <!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <title>attention 可视化</title>
            </head>
            <body>
            '''
            tail = '''
            </body>
            </html>
            '''
            self.text = head + self.text + tail
            f.write(self.text)

    def gen(self, attn_data,save_path,args):
        self.json_data = attn_data
        self.gen_html(args)
        self.save(save_path)

if __name__ == '__main__':
    fire.Fire(GenHtml)
    genHtml = GenHtml()
    attn_data = genHtml.load_json()
    genHtml.gen(attn_data, 'result/attention.html')
