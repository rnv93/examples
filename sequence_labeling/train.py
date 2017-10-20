import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import vocab
from torchtext import data
from torchtext import datasets

from model import SequenceLabeler
from util import get_args, makedirs


args = get_args()
torch.cuda.set_device(args.gpu)

print(args)

# Define columns.
WORD = data.Field(init_token="<bos>", eos_token="<eos>", lower=True)
UD_TAG = data.Field(init_token="<bos>", eos_token="<eos>")

# Build initial vocabulary.
WORD.build_vocab() # Add bos, eos and pad tokens.

special_tokens = set()
if WORD.init_token:
    special_tokens.add(WORD.vocab.stoi[WORD.init_token])
if WORD.eos_token:
    special_tokens.add(WORD.vocab.stoi[WORD.eos_token])
if WORD.pad_token:
    special_tokens.add(WORD.vocab.stoi[WORD.pad_token])

# Download and load default data.
print("Loading data")

# Ignore the third column.
train, dev, test = datasets.SequenceLabelingDataset.load_default_dataset(
    fields=(('word', WORD), ('udtag', UD_TAG), (None, None)))
print("Data loaded")

print("Loading embeddings")
# Load embeddings. Vocabulary is limited to the words in Glove.

embeddings = vocab.pretrained_aliases[args.word_vectors]()
WORD.vocab.extend(embeddings) # Build vocab from embeddings.
WORD.vocab.load_vectors(embeddings) # Copy embeddings.
del embeddings # destroy this local variable.
print("Embeddings loaded")

# Load tags
UD_TAG.build_vocab(train)

print("Creating batches")
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test), batch_size=args.batch_size, device=args.gpu)
print("Batches created")

config = args
config.n_embed = len(WORD.vocab)
config.d_out = len(UD_TAG.vocab)
config.n_cells = config.n_layers

# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot,
                       map_location=lambda storage, location: storage.cuda(
                           args.gpu))
else:
    model = SequenceLabeler(config)
    if args.word_vectors:
        model.embed.weight.data = WORD.vocab.vectors
        if args.gpu > -1:
            print("Creating CUDA model")
            model.cuda()
            print("CUDA model created")

criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
print(header)


for epoch in range(args.epochs):
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(train_iter):

        # switch model to training mode, clear gradient accumulators
        model.train(); opt.zero_grad()

        iterations += 1

        # forward pass
        answer = model(batch.word)
        predicted = torch.max(answer, 2)[1].view(-1).data
        correct = batch.udtag.view(-1).data

        # calculate accuracy of predictions in the current batch
        n_correct += (predicted == correct).sum()
        n_total += correct.size()[0]
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        loss = criterion(answer.view(-1, config.d_out), batch.udtag.view(-1))

        # backpropagate and update optimizer learning rate
        loss.backward(); opt.step()

    # evaluate performance on validation set periodically
    if epoch % args.dev_every == 0:
        dev_acc = model.evaluate(dev_iter, special_tokens)
        print(dev_log_template.format(time.time() - start,
                                      epoch, iterations, 1 + batch_idx,
                                      len(train_iter),
                                      100. * (1 + batch_idx) / len(
                                          train_iter), loss.data[0],
                                      0, train_acc,
                                      dev_acc))
    elif epoch % args.log_every == 0:

        # print progress message
        print(log_template.format(time.time()-start,
            epoch, iterations, 1+batch_idx, len(train_iter),
            100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))

    # checkpoint model periodically
    if epoch % args.save_every == 0:
        snapshot_prefix = os.path.join(args.save_path, 'snapshot')
        snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_epoch_{}_model.pt'.format(train_acc, loss.data[0], epoch)
        torch.save(model, snapshot_path)
        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)

    # update best valiation set accuracy
    if dev_acc > best_dev_acc:

        # found a model with better validation set accuracy
        best_dev_acc = dev_acc
        snapshot_prefix = os.path.join(args.save_path,
                                       'best_snapshot')
        snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__epoch_{}_model.pt'.format(
            dev_acc, 0, iterations)

        # save model, delete previous 'best_snapshot' files
        torch.save(model, snapshot_path)
        for f in glob.glob(snapshot_prefix + '*'):
            if f != snapshot_path:
                os.remove(f)

