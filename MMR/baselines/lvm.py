import pystan

lvm_code = """
data {
  int<lower = 0> N; // number of observations
  real           z[N]; // Z
  real           w[N]; // W
  real            a[N]; // A
  real           y[N]; // Y
  
}


parameters {

  vector[N] u;

  real a_uz;
  real a_uw;
  real a_ua;
  real a_uy;
  real a_za;
  real a_zy;
  real a_wy;
  
  real c_z;
  real c_w;
  real c_a;
  real c_y;
  
  
  real<lower=0> sigma_z_Sq;
  real<lower=0> sigma_w_Sq;
  real<lower=0> sigma_a_Sq;
  real<lower=0> sigma_y_Sq;
}

transformed parameters  {
 // Population standard deviation (a positive real number)
 real<lower=0> sigma_z;
 real<lower=0> sigma_w;
 real<lower=0> sigma_a;
 real<lower=0> sigma_y;
 
 // Standard deviation (derived from variance)
 sigma_z = sqrt(sigma_z_Sq);
 sigma_w = sqrt(sigma_w_Sq);
 sigma_a = sqrt(sigma_a_Sq);
 sigma_y = sqrt(sigma_y_Sq);
}

model {
  
  // don't have data about this
  u ~ normal(0, 1);
  
  // parameter priors
  // coefficients
  a_uz     ~ normal(0, 1);
  a_uw     ~ normal(0, 1);
  a_ua     ~ normal(0, 1);
  a_uy     ~ normal(0, 1);
  a_za     ~ normal(0, 1);     
  a_ay     ~ normal(0, 1);
  a_wy     ~ normal(0, 1);
 
  // intercepts
  c_z     ~ normal(0, 1);
  c_w     ~ normal(0, 1):
  c_a     ~ normal(0, 1);
  c_y     ~ normal(0, 1);
 
  // variances
  sigma_z_sq ~ inv_gamma(1, 1);
  sigma_w_sq ~ inv_gamma(1, 1);
  sigma_a_sq ~ inv_gamma(1, 1);
  sigma_y_sq ~ inv_gamma(1, 1);

  // have data about these
  z ~ normal(c_z + a_uz * u, sigma_z);
  w ~ normal(c_w + a_uw * u, sigma_w);
  a ~ normal(c_a + a_ua * u + a_za * z, sigma_a);
  y ~ normal(c_y + a_uy * u + a_ay * a + a_wy * w, sigma_y)

}
"""

stan_model = pystan.StanModel(model_code=lvm_code)
class
def fit(sm):
    sm.sampling(data=self.data, iter=1000, chains=4)

