import argparse
import xgboost as xgb
from modchooser import *
from deepevent import *
import numpy as np
import time
import pandas as pd
from osutils import *
import random
import os
from sklearn import metrics
from sklearn import manifold
from sklearn import decomposition

def load_params_boost(boost, boost_file):
	iteration = 0
	seed = random.randint(0, 2**31)
	with np.load(boost_file) as f:
		if "iteration" in f.keys():
			iteration = int(f["iteration"])
		if "seed" in f.keys():
			seed = int(f["seed"])
		boost.load_model(bytearray(f["model"].tobytes()))
	return iteration, seed

def predict_boost_model(model, data):
	preds = model.predict(data, output_margin=True)
	maxs = np.max(preds, axis=1)
	maxs = np.repeat(maxs[:, np.newaxis], len(preds[0]), axis=1)
	logits = np.exp(preds - maxs)
	lsum = np.sum(logits, axis=1)
	lsums = np.repeat(lsum[:, np.newaxis], len(logits[0]), axis=1) # for broadcasting
	return logits / lsums

def shuffle(df, axis=0):
	shuffled_df = df.copy()
	shuffled_df.apply(np.random.shuffle, axis=axis)
	return shuffled_df

def cross_entropy(preds, targets):
	res = 0.0
	for i in range(len(targets)):
		v = max(1e-6, preds[i][targets[i]])
		res -= np.log(v)
	return res

def transformBinaryCols( data ):
   for col in data.columns:
       vals = data[ col ].unique()
       if( len( vals ) == 2 ):
           if( type( vals[0] ) == type( vals[1] ) and 'int' in str( type( vals[0] ) ) ):
               data[ col ].astype( 'bool', copy = False )

def preprocess(data, args):
	if args.del_features:
		delete_features = [(22, 'ind_var2_0', [0]),(23, 'ind_var2', [0]),(57, 'ind_var27_0', [0]),(58, 'ind_var28_0', [0]),(59, 'ind_var28', [0]),(60, 'ind_var27', [0]),(81, 'ind_var41', [0]),(85, 'ind_var46_0', [0]),(86, 'ind_var46', [0]),(132, 'num_var27_0', [0]),(133, 'num_var28_0', [0]),(134, 'num_var28', [0]),(135, 'num_var27', [0]),(156, 'num_var41', [0]),(162, 'num_var46_0', [0]),(163, 'num_var46', [0]),(180, 'saldo_var28', [0]),(181, 'saldo_var27', [0]),(190, 'saldo_var41', [0]),(193, 'saldo_var46', [0]),(221, 'imp_amort_var18_hace3', [0]),(223, 'imp_amort_var34_hace3', [0]),(235, 'imp_reemb_var13_hace3', [0]),(239, 'imp_reemb_var33_hace3', [0]),(245, 'imp_trasp_var17_out_hace3', [0]),(249, 'imp_trasp_var33_out_hace3', [0]),(262, 'num_var2_0_ult1', [0]),(263, 'num_var2_ult1', [0]),(304, 'num_reemb_var13_hace3', [0]),(308, 'num_reemb_var33_hace3', [0]),(316, 'num_trasp_var17_out_hace3', [0]),(320, 'num_trasp_var33_out_hace3', [0]),(328, 'saldo_var2_ult1', [0]),(350, 'saldo_medio_var13_medio_hace3', [0])]
		for _, name, _ in delete_features:
			data.drop(labels=name, inplace=True, axis=1)
	if args.var3_nan:
		data['var3_NaN'] = (data['var3'] == -999999)
	if args.tr_bin:
		transformBinaryCols(data)
	if args.drop_id:
		data.drop(labels="ID", axis=1, inplace=True)
	if args.pca:
		res = decomposition.PCA(2).fit_transform(data)
		data["pca_0"] = res[:, 0]
		data["pca_1"] = res[:, 1]

def preprocess_flags(parser):
	parser.add_argument("--del-features", dest="del_features", default=False, action="store_true", help="1")
	parser.add_argument("--var3-nan", dest="var3_nan", default=False, action="store_true", help="2")
	parser.add_argument("--tr-bin", dest="tr_bin", default=False, action="store_true", help="3")
	parser.add_argument("--drop-id", dest="drop_id", default=False, action="store_true", help="5")
	parser.add_argument("--pca", default=False, action="store_true", help="6")

def main_train(args):
	parser = argparse.ArgumentParser(description="Train a tree boost model.")
	parser.add_argument("--data", required=True, nargs=2, help="dataset to train on")
	parser.add_argument("--iters", type=int, default=500, help="number of iterations to train")
	parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=0.01, help="learning rate for SGD")
	parser.add_argument("--file", help="three model file")
	parser.add_argument("--out", required=True, help="output network file prefix")
	parser.add_argument("--deepevent", nargs=2, help="url to deepevent service")
	parser.add_argument("--threads", type=int, default=available_cpu_count(), help="number of threads")
	parser.add_argument("--sample", type=float, nargs=2, default=[0.5, 0.5], help="subsampling settings (samples, features)")
	parser.add_argument("--lambda", dest="plambda", type=float, default=1, help="lambda parameter for booster")
	parser.add_argument("--depth", type=int, default=6, help="tree depth parameter for booster")
	parser.add_argument("--auc", default=False, action="store_true", help="calc auc instead of logloss")
	parser.add_argument("--no-sample", dest="no_sample", default=False, action="store_true", help="no sample")
	preprocess_flags(parser)
	args = parser.parse_args(args)
	print(args)

	mkdir_p(os.path.dirname(args.out))

	if args.deepevent:
		deepevent = DeepEvent(args.deepevent[0], args.deepevent[1])
	else:
		deepevent = DeepEventDummy()

	learning_rate_evt = StepScalarEventCreator("Learning rate")
	epoch_time_evt = StepScalarEventCreator("Epoch time")
	train_loss_evt = StepScalarEventCreator("Losses", "Training loss")
	valid_loss_evt = StepScalarEventCreator("Losses", "Validation loss")
	valid_acc_evt = StepScalarEventCreator("Accuracy", "Validation accuracy")

	params = {
		"silent": 1,
		"nthread": args.threads,
		# "bst:max_depth": args.depth,
		"bst:subsample": args.sample[0],
		"bst:colsample_bytree": args.sample[1],
		# "bst:lambda": args.plambda,
		"objective": "multi:softmax",
		"eval_metric": "auc",
		"num_class": 2,
		"bst:eta": 0.01,
	}

	model = xgb.Booster(params)
	iteration = 0
	seed = random.randint(0, 2**31)
	if args.file and os.path.exists(args.file):
		iteration, seed = load_params_boost(model, args.file)

	train_data = pd.read_csv(args.data[0], header=0)

	train_data = train_data.sample(n=len(train_data), random_state=seed)
	preprocess(train_data, args)

	train_x = train_data.drop(labels="TARGET", axis=1)
	train_y = train_data["TARGET"]

	if not args.no_sample:
		train_split = int(len(train_x) * 0.75)
		val_x = train_x[:train_split]
		val_y = train_y[:train_split]
		train_x = train_x[train_split:]
		train_y = train_y[train_split:]

	print("Starting training...")

	start_time = time.time()
	train_size = len(train_y)
	if not args.no_sample:
		data = xgb.DMatrix(val_x, label=val_y)
		val_predictions = predict_boost_model(model, data)
		if args.auc:
			val_err = metrics.roc_auc_score(np.array(val_y), val_predictions[:, 1], average="weighted") * len(val_y)
		else:
			val_err = cross_entropy(val_predictions, np.array(val_y))
		val_acc = (np.argmax(val_predictions, axis=1) == np.array(val_y)).sum()
		best_val_err = val_err
		test_size = len(val_y)
		iter_time = time.time() - start_time
		print("Iteration {} of {} took {:.3f}s".format(iteration, args.iters, iter_time))
		print("  validation loss:\t\t{:.6f}".format(val_err / test_size))
		print("  validation accuracy:\t\t{:.3f} %".format(val_acc / test_size * 100.0))

		deepevent.send(learning_rate_evt.new(iteration, args.learning_rate))
		deepevent.send(valid_loss_evt.new(iteration, val_err / test_size))
		deepevent.send(valid_acc_evt.new(iteration, val_acc / test_size * 100.0))

	# There is a bug in xgboost: xgb.train segfaults when xgb_model is not a trained model, but just a xgb.Booster(params).
	if iteration == 0:
		model = None

	for iteration in range(iteration + 1, iteration + 1 + args.iters):
		start_time = time.time()

		data = xgb.DMatrix(train_x, label=train_y)
		model = xgb.train(
			params=params,
			dtrain=data,
			num_boost_round=1,
			learning_rates=[args.learning_rate],
			verbose_eval=False,
			xgb_model=model,
		)
		train_predictions = predict_boost_model(model, data)
		if args.auc:
			train_err = metrics.roc_auc_score(np.array(train_y), train_predictions[:, 1], average="weighted") * len(train_y)
		else:
			train_err = cross_entropy(train_predictions, np.array(train_y))

		if not args.no_sample:
			data = xgb.DMatrix(val_x, label=val_y)
			val_predictions = predict_boost_model(model, data)
			if args.auc:
				val_err = metrics.roc_auc_score(np.array(val_y), val_predictions[:, 1], average="weighted") * len(val_y)
			else:
				val_err = cross_entropy(val_predictions, np.array(val_y))
			val_acc = (np.argmax(val_predictions, axis=1) == np.array(val_y)).sum()

		iter_time = time.time() - start_time
		print("Iteration {} of {} took {:.3f}s".format(iteration, args.iters, iter_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_size))
		if not args.no_sample:
			print("  validation loss:\t\t{:.6f}".format(val_err / test_size))
			print("  validation accuracy:\t\t{:.3f} %".format(val_acc / test_size * 100.0))

		deepevent.send(train_loss_evt.new(iteration, train_err / train_size))
		if not args.no_sample:
			deepevent.send(valid_loss_evt.new(iteration, val_err / test_size))
			deepevent.send(valid_acc_evt.new(iteration, val_acc / test_size * 100.0))
		deepevent.send(epoch_time_evt.new(iteration, iter_time))

		np.savez(args.out + "-%d" % iteration, iteration=iteration, seed=seed, model=np.array(model.save_raw(), dtype=np.uint8))

		if not args.no_sample:
			if val_err < best_val_err:
				best_val_err = val_err
				np.savez(args.out + "-best", iteration=iteration, seed=seed, model=np.array(model.save_raw(), dtype=np.uint8))

def main_test(args):
	parser = argparse.ArgumentParser(description="Train a tree boost model.")
	parser.add_argument("--data", required=True, help="dataset to train on")
	parser.add_argument("--file", required=True, help="three model file")
	parser.add_argument("--out", required=True, help="output network file prefix")
	preprocess_flags(parser)
	args = parser.parse_args(args)
	print(args)

	model = xgb.Booster()
	if args.file and os.path.exists(args.file):
		iteration, seed = load_params_boost(model, args.file)

	test_data = pd.read_csv(args.data, header=0)
	ids = test_data["ID"]
	preprocess(test_data, args)

	val_predictions = predict_boost_model(model, xgb.DMatrix(test_data))
	answers = val_predictions[:, 1]
	res = open(args.out, "w")
	res.write("ID,TARGET\n")
	n = 0
	for a in answers:
		res.write(str(ids[n]) + "," + str(a) + "\n")
		n += 1
	res.close()

def main_tsne(args):
	parser = argparse.ArgumentParser(description="Train a tree boost model.")
	parser.add_argument("--data", required=True, nargs=2, help="dataset to train on")
	parser.add_argument("--out", required=True, help="output network file prefix")
	preprocess_flags(parser)
	args = parser.parse_args(args)
	print(args)

	train_data = pd.read_csv(args.data[0], header=0)
	train_data.drop(labels="TARGET", axis=1, inplace=True)
	test_data = pd.read_csv(args.data[1], header=0)
	preprocess(test_data, args)
	preprocess(train_data, args)


	print(res.shape)

if __name__ == '__main__':
	(ModChooser("Deep learning tools.")
		.add("train", main_train, "train network")
		.add("test", main_test, "test boost")
		.add("tsne", main_tsne, "tsne")
		.main())
