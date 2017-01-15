import numpy as np
from q2_NER import get_hyperparams_config, hyperband_train_and_validate

max_iter = 27 # maximum iterations/epochs per configuration
eta = 3 # defines downsampling rate (default=2)
logeta = lambda x: np.log(x)/np.log(eta)
s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

#### Begin Finite Horizon Hyperband outlerloop. Repeat indefinetely.
losses_dict = dict()
best_config = None
best_val_loss = float('inf')
for s in reversed(range(s_max+1)):
	n = int(np.ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
	r = max_iter*eta**(-s) # initial number of iterations to run configurations for

	#### Begin Finite Horizon Successive Halving with (n,r)
	T = [ get_hyperparams_config() for i in range(n) ] 
	for i in range(s+1):
		# Run each of the n_i configs for r_i iterations and keep best n_i/eta
		n_i = n*eta**(-i)
		r_i = int(np.floor(r*eta**(i)))
                print('number of configurations: ' + str(n_i) + ', epochs to run: ' + str(r_i))
		val_losses = [ hyperband_train_and_validate(r_i, t) for t in T ]

                indices = np.argsort(val_losses)
                idx = indices[0]
                curr_val_loss = val_losses[idx]
                curr_best_config = T[idx]

		losses_dict = dict(zip(T, val_losses))
		T = [ T[i] for i in indices[0:int( n_i/eta )] ]

        if(curr_val_loss < best_val_loss):
                best_val_loss = curr_val_loss
                best_config = curr_best_config
	#### End Finite Horizon Successive Halving with (n,r)

target = open('losses_dict', 'w')
for key, value in losses_dict.iteritems():
    target.write(str(key)+': ' + str(value) + '\n')
target.close()

print(best_val_loss)
print(best_config)
