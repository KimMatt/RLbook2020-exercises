## Exercise 11.1
*Convert the equation of n-step off-policy TD (7.9) to semi-gradient form.
Give accompanying definitions of the return for both the episodic and continuing cases.*

$w_{t+n} = w_{t+n-1} + \alpha \rho_{t+1} ... \rho_{t+n-1} \delta_t \triangledown \hat{v}(S_t,w_{t+n-1})$

episodic case

$\delta_t = R_{t+1} + ... + \gamma^{n-1}R_{t+n} + \gamma^n \hat{v}(S_{t+n},w_{t+n-1}) - \hat{v}(S_t, w_{t+n-1})$

continuing case

$\delta_t = R_{t+1} - \bar{R}_t ... + R_{t+n} - \bar{R}_{t+n-1} + \hat{v}(S_{t+n},w_{t+n-1}) - \hat{v}(S_t, w_{t+n-1})$


## Exercise 11.2
*Convert the equations of n-step Q() (7.11 and 7.17) to semi-gradient
form. Give definitions that cover both the episodic and continuing cases.*


