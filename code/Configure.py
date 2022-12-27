# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/wrn_v2/',
	"depth": 28,
	"layers_per_block": 4,
	"num_classes": 10,
	"width_multiplier": 10,
	"dropout_rate": 0.3
}

training_configs = {
	"learning_rate": 0.1,
	"weight_decay": 5e-4,
	"momentum": 0.9,
	"save_interval": 10,
	'batch_size': 128,
	'max_epoch': 250,
}

### END CODE HERE