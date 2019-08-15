import json
import fire
class GenHtml(object):
    def __init__(self):
        self.json_path = 'attn_data.json'

    def load_json(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.json_data = json.load(f)
        return self.json_data

    def highlight(self, word, attn):
        html_color = '#%02X%02X%02X' % (255, int(255 * (1 - attn)), int(255 * (1 - attn)))
        return '<span style="background-color: {}">{}</span>'.format(html_color, word)

    def mk_html(self, seq, attns):
        html = ""
        attns = attns[0:len(seq)]
        for ix, attn in zip(seq, attns):
            html += ' ' + self.highlight(ix, attn)
        return '<li>' + html + '</li>'

    def mk_label(self,label,pre_label):
        html = ''
        html = html + '<span>' + "真实标签：" + str(label) + '</span>' + " | "
        html = html + '<span>' + "预测标签：" + str(pre_label) + '</span>'
        return '<li>' + html + '</li>'

    def gen_html(self):
        seqs = self.json_data['sequences']
        attns = self.json_data['attention_weights']
        label = self.json_data['rea_labels']
        pre_label = self.json_data['pre_labels']
        batch_size = len(label)
        text = '<ul>'
        for i in range(batch_size):
            text += self.mk_label(label[i],pre_label[i])
            for j in range(3):
                text += self.mk_html(seqs[i], attns[i*3+j])
        text += '</ul>'
        self.text = text

    def save(self,weight_type,save_path):
        with open(save_path+weight_type+'.html', 'w', encoding='utf-8') as f:
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

    def gen(self, attn_data, weight_type, save_path):
        self.json_data = attn_data
        self.gen_html()
        self.save(weight_type,save_path)

if __name__ == '__main__':
    fire.Fire(GenHtml)










