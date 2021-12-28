import time, re, itertools, random, subprocess, traceback
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from tqdm import tqdm

from data.err_dataset import ErrDataset
from model import CRmodel
from utils import Stats, try_gpu

from code_process import tokenize_code
from code_process import fix_strings
from code_process import anonymize_code_str, deanonymize_code_str, tokenize_err_msg


class Experiment(object):
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.dataset = ErrDataset()
        self.create_model()
        self.model.initialize()

    def close(self):
        pass

    def create_model(self):
        self.model = CRmodel(self.dataset.vocab_r, self.dataset.vocab_x)
        self.model = try_gpu(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=0)

    def train(self):
        state_dict = self.model.state_dict()
        torch.save(state_dict, './train/trained_model.model')

        max_steps = 150000
        self.step = 0
        progress_bar = tqdm(total=max_steps, desc='TRAIN', mininterval=20)
        progress_bar.update(self.step)

        train_iter = None
        train_stats = Stats()

        while self.step < max_steps:
            self.step += 1
            progress_bar.update()

            train_batch = None if train_iter is None else next(train_iter, None)
            if train_batch is None:
                self.dataset.init_iter('train')
                train_iter = self.dataset.get_iter('train')
                train_batch = next(train_iter)

            stats = self.process_batch(train_batch, train=True)
            train_stats.add(stats)

            if self.step % 50 == 0:
                print('TRAIN @ {}: {}'.format(self.step, train_stats))
                train_stats = Stats()

            if self.step % 5000 == 0 or self.step == max_steps:
                state_dict = self.model.state_dict()
                torch.save(state_dict, './train/trained_model.model')

            # if self.step % 150000 == 0:
            #     dev_stats = Stats()
            #     self.dataset.init_iter('dev')
            #     fout = open('./train/pred.dev.{}'.format(self.step), 'w')
            #     for dev_batch in tqdm(self.dataset.get_iter('dev'), desc='DEV', mininterval=60):
            #         stats = self.process_batch(dev_batch, train=False, fout=fout)
            #         dev_stats.add(stats)
            #     if fout: fout.close()
            #     print('DEV @ {}: {}'.format(self.step, dev_stats))
        progress_bar.close()

    def test(self):

        START_TIME = time.time()

        def decide_is_directive(code_line):
            code_line = code_line.strip()
            if re.match(r'#\w+', code_line): return True
            if re.sub(r"\s", "", code_line) == "usingnamespacestd;": return True
            return False

        def prepare_code_just_as_it_is(pred_stmt, code_lines_str):
            assert len(pred_stmt) == len(code_lines_str)
            ret_lines = []
            code_lines = []
            anonymize_dicts = []
            for j, line in enumerate(code_lines_str):
                line = fix_strings(line, only_char=True)
                curr_line_for_repair_model, anonymize_dict = anonymize_code_str(line)
                ret_lines.append(line)
                code_lines.append((pred_stmt[j]["text"], curr_line_for_repair_model, pred_stmt[j]["indent"]))
                anonymize_dicts.append(anonymize_dict)
            ret_code = "\n".join(ret_lines)
            return ret_code, code_lines, anonymize_dicts

        def prepare_lines_print(text_indent, text_str_noindt, _max_len, wrap_indent=3):
            text_str = text_indent * "  " + text_str_noindt
            text_to_print = []
            if len(text_str) <= _max_len:
                text_to_print.append(text_str)
            else:
                text_str_print = text_str[:_max_len]
                text_to_print.append(text_str_print)
                text_str_noindt = text_str[_max_len:]
                text_indent += wrap_indent
                text_str = text_indent * "  " + text_str_noindt
                while len(text_str) > _max_len:
                    text_str_print = text_str[:_max_len]
                    text_to_print.append(text_str_print)
                    text_str_noindt = text_str[_max_len:]
                    text_str = text_indent * "  " + text_str_noindt
                text_to_print.append(text_str)
            return text_to_print

        def parse_error(raw_err_msg, line_offset, tokenize=True):
            lines = raw_err_msg.split('\n')
            for line in lines:
                m = re.match('[^:]*:(\d+):[:0-9 ]+error: (.*)', line)
                if not m:
                    continue
                lineno, message = m.groups()
                if tokenize:
                    message = ' '.join(tokenize_err_msg(message))
                return int(lineno) - line_offset, message.strip()

            for line in lines:
                m = re.match('[^:]*:(\d+):[:0-9 ]+: (.*)', line)
                if not m:
                    continue
                lineno, message = m.groups()
                if tokenize:
                    message = ' '.join(tokenize_err_msg(message))
                return int(lineno) - line_offset, message.strip()

            return None, None

        inp_stmt, pred_stmt = [], []

        with open('./test/test.cpp') as src_in:
            for raw_line in src_in:
                _code = raw_line.strip()
                _is_directive = decide_is_directive(raw_line)
                _code_str_tokenized = ' '.join(tokenize_code(_code, mod_brace=False))
                _indent = max(len(re.match('^ *', raw_line).group(0)) // 4, len(re.match('^\t*', raw_line).group(0)))
                inp = [
                    "",  # text
                    _code,  # code
                    len(pred_stmt),  # line
                    _indent  # indent
                ]
                pred = {
                    'line': len(pred_stmt),
                    'text': "DUMMY" if _is_directive else "",
                    'code': _code if _is_directive else _code_str_tokenized,
                    'indent': _indent,
                }
                inp_stmt.append(inp)
                pred_stmt.append(pred)

            pred_stmt = pred_stmt[:]
            curr_code_lines_str = []
            for stmt_idx, (inp, pred) in enumerate(zip(inp_stmt, pred_stmt)):
                if pred["text"] != 'DUMMY':
                    curr_code_lines_str.append(pred["code"])
                else:
                    curr_code_lines_str.append(pred["code"])

            self.model.load_state_dict(torch.load('./train/trained_model.model'))

            iter_count = 0
            budget = 5

            with open('./test/result.txt', 'w') as stat_file:
                while iter_count < budget:
                    stat_file.flush()
                    iter_count += 1
                    stat_file.write("Iteration # " + str(iter_count) + "\n")
                    stat_file.write("Time: {:.3f}\n".format(time.time() - START_TIME))

                    code, code_lines, anonymize_dicts = prepare_code_just_as_it_is(pred_stmt, curr_code_lines_str)
                    print("Current code:", file=stat_file)
                    raw_code_lines = code.split("\n")
                    for (lineno, code_line) in enumerate(code_lines):
                        code_str = code_line[1]
                        text_str = raw_code_lines[lineno]
                        indent = code_line[2]

                        gold_to_print = []
                        if gold_to_print == []: gold_to_print.append("")
                        text_to_print = prepare_lines_print(indent, text_str, 50)
                        code_to_print = prepare_lines_print(indent, code_str, 50)
                        _text = text_to_print.pop(0)
                        _code = code_to_print.pop(0)
                        _gold = gold_to_print.pop(0)
                        print(
                            "{:>3}  {:<{width1}}  {:<{width2}}  {:}".format(str(lineno), _text, _code, _gold, width1=50,
                                                                            width2=50), file=stat_file)
                        for _text, _code, _gold in itertools.zip_longest(text_to_print, code_to_print, gold_to_print):
                            if _text is None: _text = ""
                            if _code is None: _code = ""
                            if _gold is None: _gold = ""
                            print("{:>3}  {:<{width1}}  {:<{width2}}  {:}".format("", _text, _code, _gold, width1=50,
                                                                                  width2=50), file=stat_file)
                    with open('./test/fixed.cpp', "w") as src_file:
                        src_file.flush()
                        src_file.write(code)
                    command = "gcc -w -std=c99 -pedantic ./test/fixed.cpp -lm -o ./test/fixed"
                    process = subprocess.run(command, shell=True, timeout=10, stderr=subprocess.PIPE)

                    if process.returncode == 0:
                        stat_file.write("compiled!\n\n")
                        break
                    else:
                        process = process.stderr.decode('utf8', 'backslashreplace')
                        curr_num_of_compiler_errs = len(process.split("\n"))
                        try:
                            raw_compiler_err_msg = ['\n'.join(x.replace('./test/fixed.cpp', '')
                                                              for x in process.split('\n')
                                                              if x.startswith('./test/fixed'))]
                            lineno, msg = parse_error(process, 1, tokenize=True)
                            err_line_obj = {'lineno': lineno, 'msg': msg}
                            q = {
                                'info': 'N/A',
                                'code_lines': code_lines,
                                'err_line': err_line_obj,
                                "gold_linenos": [0]*len(code_lines),
                                "edit_linenos": [0]*len(code_lines),
                                "gold_code_lines": code_lines,
                                "comment": {"method": "localize_only", "beam_size": 10}
                            }
                            batch, comment = self.dataset.s_parse_request(q)
                            self.model.eval()
                            result_stuff = self.model.forward_encode(batch)
                            # localize
                            logit_localize, label_localize = self.model.forward_localize(batch, result_stuff)
                            pred_localize = self.model.get_pred_localization(logit_localize, batch)
                            pred_lineno = pred_localize[0].item()  # one scalar
                            # edit
                            logit_edit, label_edit = self.model.forward_edit(batch, result_stuff, train_mode=False,
                                                                            beam_size=comment["beam_size"],
                                                                            edit_lineno_specified=[pred_lineno])
                            pred_edit = self.model.get_pred_edit(logit_edit, batch, train_mode=False, retAllHyp=True)

                            response = self.dataset.s_generate_response(q, batch, logit_localize, (pred_localize, pred_edit))
                            pred_code_nbest = response["pred_edit"]
                        except Exception as e:
                            print('PANIC! {}'.format(e))
                            print('PANIC! {}'.format(e), file=stat_file)
                            stat_file.write(traceback.format_exc())
                            continue

                        ##try to update the code
                        print("compiler err msg  : %s" % (err_line_obj["msg"]), file=stat_file)
                        print("compiler err line#: %d" % (err_line_obj["lineno"]), file=stat_file)
                        print("pred err line#: %d" % (pred_lineno), file=stat_file)

                        accept = False
                        for pred_code in pred_code_nbest:
                            pred_code_deano = deanonymize_code_str(pred_code, anonymize_dicts[pred_lineno])
                            if pred_code_deano is None:
                                continue
                            print("pred_code_candidate:", pred_code_deano, file=stat_file)
                            tmp_curr_code_lines_str = curr_code_lines_str[:]
                            tmp_curr_code_lines_str[pred_lineno] = pred_code_deano
                            tmp_code, _, _ = prepare_code_just_as_it_is(pred_stmt, tmp_curr_code_lines_str)
                            with open('./test/fixed.cpp', "w") as src_file:
                                src_file.flush()
                                src_file.write(tmp_code)
                            command = "gcc -w -std=c99 -pedantic ./test/fixed.cpp -lm -o ./test/fixed"
                            tmp_error_message = subprocess.run(command, shell=True, timeout=10, stderr=subprocess.PIPE)
                            if tmp_error_message == None or len(tmp_error_message.stderr.decode('utf8', 'backslashreplace').split("\n")) < curr_num_of_compiler_errs:
                                accept = True
                                break

                        if not accept:
                            print("all edit candidates rejected.", file=stat_file)
                            return False, False

                        print("pred code (edit): %s" % (pred_code_deano), file=stat_file)
                        curr_code_lines_str[pred_lineno] = pred_code_deano
                        stat_file.write("\n\n")
                        continue

    def process_batch(self, batch, train=False, fout=None):
        stats = Stats()

        if train:
            self.optimizer.zero_grad()
            self.model.train()
            all_enc_stuff = self.model.forward_encode(batch)
            logit_localize, label_localize = self.model.forward_localize(batch, all_enc_stuff)
            logit_edit, label_edit = self.model.forward_edit(batch, all_enc_stuff, train_mode=0.5)

            loss_localize = self.model.get_loss_localization(logit_localize, label_localize, batch)
            loss_edit = self.model.get_loss_edit(logit_edit, label_edit, batch)

            loss_localize = loss_localize.sum() / len(batch)
            loss_edit = loss_edit.sum() / len(batch)
            mean_loss = 0.5 * (loss_localize + loss_edit)

            stats.n = len(batch)
            stats.n_batches = 1
            stats.loss_localize = float(loss_localize)
            stats.loss_edit = float(loss_edit)
            # Evaluate
            pred_localize = self.model.get_pred_localization(logit_localize, batch)
            pred_edit = self.model.get_pred_edit(logit_edit, batch)
            self.dataset.evaluate(batch, [logit_localize, logit_edit, None], [pred_localize, pred_edit, None], stats, fout)
            # Gradient
            if mean_loss.requires_grad:
                mean_loss.backward()
                stats.grad_norm = clip_grad_norm_(
                    self.model.parameters(),
                    10
                )
                self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():
                all_enc_stuff = self.model.forward_encode(batch)
                # localize
                logit_localize, label_localize = self.model.forward_localize(batch, all_enc_stuff)
                pred_localize = self.model.get_pred_localization(logit_localize, batch)
                pred_lineno = pred_localize[0].item()

                # edit
                logit_edit1, label_edit1 = self.model.forward_edit(batch, all_enc_stuff, train_mode=False, beam_size=10)
                logit_edit2, label_edit2 = self.model.forward_edit(batch, all_enc_stuff, train_mode=False, beam_size=10,
                                                                   edit_lineno_specified=[pred_lineno])
                pred_edit1 = self.model.get_pred_edit(logit_edit1, batch, train_mode=False)
                pred_edit2 = self.model.get_pred_edit(logit_edit2, batch, train_mode=False)

                stats.n = len(batch)
                stats.n_batches = 1
                stats.loss_localize = 0.
                stats.loss_edit = 0.
                logit_edit1 = torch.zeros([stats.n, 1])
                logit_edit2 = torch.zeros([stats.n, 1])

                # Evaluate
                self.dataset.evaluate(batch, [logit_localize, logit_edit1, logit_edit2], [pred_localize, pred_edit1, pred_edit2], stats, fout)

        return stats
