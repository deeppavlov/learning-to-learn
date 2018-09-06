from learning_to_learn.environment import Environment
from learning_to_learn.pupils.lstm_for_meta import Lstm, LstmFastBatchGenerator as BatchGenerator


voc_name = 'dost_voc.txt'
# voc_name = 'text8_voc.txt'
with open(voc_name, 'r') as f:
    vocabulary = list(f.read())

vocabulary_size = len(vocabulary)

env = Environment(Lstm, BatchGenerator, vocabulary=vocabulary)

valid_add_feed = [{'placeholder': 'dropout', 'value': 1.}]

env.build_pupil(
    num_layers=2,
    num_nodes=[1500, 1500],
    num_output_layers=1,
    num_output_nodes=[],
    vocabulary_size=vocabulary_size,
    embedding_size=500,
    num_gpus=1,
    regime='inference',
)


# env.build_pupil(
#     num_layers=1,
#     num_nodes=[100],
#     num_output_layers=1,
#     num_output_nodes=[],
#     vocabulary_size=vocabulary_size,
#     embedding_size=150,
#     num_gpus=1,
#     regime='inference',
# )


def continue_dialog(fuse, n):
    fuse = dict(
        text=fuse,
        num_repeats=1,
        max_num_of_chars=500,
        fuse_stop='limit',
    )

    fuse_res, _ = env.test(
        allow_growth=True,
        restore_path='dostoevsky/train/checkpoints/best',
        # restore_path='lstm/start/checkpoints/best',
        print_results=False,
        verbose=False,
        fuses=[fuse],
        vocabulary=vocabulary,
        additions_to_feed_dict=valid_add_feed,
    )

    dialog_continuation = fuse_res[0]['results'][0].split('\n')
    dialog_continuation = [replica for replica in dialog_continuation if len(replica) > 0]
    return dialog_continuation[:n]

if __name__ == '__main__':
    print(continue_dialog('- Ты кто?\n', 10))