# GraphSAM


## Training

###CoMPT <br>
cd CoMPT-main <br>
``
python train_graph.py --seed 777 --fold 5 --epochs 30 --dataset <datasets> --split random --sam <GraphSAM> --rho <rho> --radius <radius> --epoch_steps <lambda> --alpha <beta> --gamma <gamma>
``
<br>
where `` <rho> `` is neighborhood ball size. `` <GraphSAM>=2 `` is the GraphSAM algorithm. 
`` <radius> `` is generally equal to `` <rho> ``. `` <lambda> `` is rho's update rate .  `` <gamma> ``is step_rho's learning rate. `` <beta> `` is smoothing parameters of moving average.
<br>
``
python ns_graph.py --epochs <epochs> --par <par> --rate <rate>
``

```python
from GraphSAM import GraphSAM

model = YourModel()
criterion = YourCriterion()
base_optimizer = YourBaseOptimizer

optimizer = GraphSAM(
    params=model.parameters(),
    rho=0.05,
    beta=0.99,
    gamma=0.5,
    lambda=1,
    base_optimizer=base_optimizer,
    **kwargs
)
for epoch in range(epochs):
	i=0
	for batch in (train_loader):
		def closure():
			output = model(input)
			loss = loss_f(output, target)
			loss.backward()
			return loss

		if i==0:
			output = model(input)
			loss=loss_f(output,target)
			optimizer.step(i, epoch, closure, loss)
		else:
			optimizer.step(i, epoch, closure)
		loss = optimizer.get_loss()
		optimizer.zero_grad()
...

    ```
