function [err,m_est,xeopt] = optMOGP(num_kernel,in_dim,out_dim,x0,kernel_type_str,training_input,...
  obs,test_input,gt,plot_traj)
  % Optimize and predict
  % Parameters:
  %   num_kernel (integer)
  %     number of kernel functions
  %   in_dim (integer)
  %     dimension of inputs required in each kernel
  %   out_dim (integer)
  %     dimension of outputs
  %   x0 (1 x n_param vector)
  %     initial hyperparameters 
  %   kernel_type_str (string)
  %     type of kernels used for multioutput Gaussian processes
  %   training_points (in_dim x n matrix)
  %     each column represents one of the training inputs
  %   obs (1 x n*out_dim vector)
  %     observations at training points, a vector of all elements of an out_dim x n 
  %     matrix, taken row by row
  %   test_points (in_dim x m matrix)
  %     each column represents one of the test inputs
  %   gt (out_dim x m matrix)
  %     groundtruth at test points 
  %   plot_traj (bool)
  %     decide to plot trajectory or not
  % Returns:  
  %   err (scalar)  
  %     root mean squared error (RMSE) of estimates

  % compute number of hyperparameters and get function handle for each kernel
  kernel_type = cell(1,num_kernel);
  kernel_type(:) = kernel_type_str;
  fh_cell = cell(1,num_kernel);
  num_hyp_k = 1;
  for j = 1:num_kernel
    fh = str2func(kernel_type{j});
    fh_cell{j} = fh;
    num_hyp_k = num_hyp_k + fh(in_dim);
  end

  % define manifolds of hyperparameters
  if ~strcmp(kernel_type{j},'multiVariateGeneralizedvMKernel')
    % a vector of positive hyperparameters is needes for most kernels
    elements.kernParam = positivefactory(num_hyp_k,1);
  else
    elements.kernParam = positivefactory(1,1);
    % a positive definite matrix is required for mGvM kernel 
    elements.kernParam1 = sympositivedefinitefactory(2*in_dim);
  end
  elements.icmParam = sympositivedefinitefactory(out_dim);
  elements.noiseParam = positivefactory(out_dim,1);
  problem.M = productmanifold(elements);
  
  % initialize hyperparameters
  if ~strcmp(kernel_type{j},'multiVariateGeneralizedvMKernel')
    x1.kernParam = x0(1:num_hyp_k)';
  else
    x1.kernParam1 = reshape(x0(2:num_hyp_k),2*in_dim,2*in_dim);
    x1.kernParam = x0(1);
  end  
  x1.icmParam = reshape(x0(num_hyp_k+1:num_hyp_k+out_dim*out_dim),out_dim,out_dim);
  x1.noiseParam = x0(end-out_dim+1:end)';

  options.verbosity = 0;
  options.maxiter = 100;
  warning('off', 'manopt:getHessian:approx');

  % define cost function and gradient
  problem.cost = @(x) objfcn(x,training_input,obs,fh_cell,in_dim,out_dim);
  problem.egrad = @(x) egradMo(x,training_input,obs,fh_cell,in_dim,out_dim);

  % optimize
  tic;
  [xe] = trustregions(problem,x1,options);
  toc;
  
  % compute predictive distributions at test points
  xeopt = x2xopt(xe);
  
  covfunc = @(x1,x2) icm(fh_cell,xeopt,x1,x2,in_dim,out_dim);
  gaussp = MultiOutputGaussianProcess(covfunc,out_dim);
  [m_est,covar] = gaussp.predict(training_input,obs,xe.noiseParam,test_input);

  % evaluate RMSE
  n_test = size(test_input,2);
  m_est_array = reshape(m_est,n_test,out_dim)';
  err = sqrt(sum(sum((gt - m_est_array).^2))/size(gt,2));
  fprintf(['RMSE of estimation using ' kernel_type{1} ' : %f\n'],err);

  % plot trajectory and covariances
  if plot_traj
    visTrajandCov(m_est(1:n_test),m_est(1+n_test:2*n_test),m_est(1+2*n_test:3*n_test),gt,...
      obs,covar,size(test_input,2),kernel_type_str{1});       
  end
end    

function [f] = objfcn(x,training_input,obs,fh_cell,input_dim,output_dim)
  % Objective function for the optimization problem
  % Parameters:
  %   x (struct)
  %     hyperparameters
  %   training_points (in_dim x n matrix)
  %     each column represents one of the training inputs
  %   obs (1 x n*out_dim matrix)
  %     observations at training points
  %   fh_cell (cell)
  %     cell of function handles for kernel functions
  %   input_dim (integer)
  %     dimension of inputs required in each kernel
  %   output_dim (integer)
  %     dimension of outputs required in each kernel
  % Returns:
  %   f (scalar)
  %     cost of the objective function 

  cov = x.noiseParam;
  xopt = x2xopt(x);
  covfunc = @(x1,x2) icm(fh_cell,xopt,x1,x2,input_dim,output_dim);
  gaussp = MultiOutputGaussianProcess(covfunc,output_dim);
  f = gaussp.computeLogMarginalLik(training_input,obs,cov);
end   

function [egopt] = egradMo(x,training_input,obs,fh_cell,input_dim,output_dim)
  % Gradients of the objective function w.r.t. hyperparameters in Euclidean spaces
  % Parameters:
  %   training_points (in_dim x n matrix)
  %     each column represents one of the training inputs
  %   obs (1 x n*out_dim matrix)
  %     observations at training points  
  %   fh_cell (cell array)
  %     cell of function handles
  %   in_dim (integer)
  %     dimension of inputs required in each kernel
  %   out_dim (integer)
  %     dimension of outputs  
  % Returns:
  %   egopt (struct)
  %     struct of gradients of the objective function in Euclidean spaces 
  %     w.r.t. hyperparameters 

  cov = x.noiseParam;
  xopt = x2xopt(x);
  covfunc = @(x1,x2) icm(fh_cell,xopt,x1,x2,input_dim,output_dim);
  gaussp = MultiOutputGaussianProcess(covfunc,output_dim);
  [~,eg] = gaussp.computeLogMarginalLik(training_input,obs,cov);
  egopt = eg2egopt(x,eg);
end   

function xopt = x2xopt(x)
  % Convert hyperparameters to compromise different manifolds required for different kernels
  % Parameters:
  %   x (struct)
  %     struct of hyperparameters for evaluating different kernels
  % Returns:
  %   xopt (struct)
  %     struct of hyperparameters for optimization

  xopt.noiseParam = x.noiseParam;
  xopt.icmParam = x.icmParam;
  if ~isfield(x,'kernParam1')
    xopt.kernParam = x.kernParam;
  else
    xopt.kernParam = [x.kernParam;x.kernParam1(:)];
  end  
end

function egopt = eg2egopt(x,eg)
  % Convert hyperparameters to compromise different manifolds required for different kernels
  % Parameters:
  %   eg (struct)
  %     struct of gradients of negative log likelihood computed from MOGP
  % Returns:
  %   egopt (struct)
  %     struct of gradients of negative log likelihood  for optimization
  
  egopt.noiseParam = eg.noiseParam;
  egopt.icmParam = eg.icmParam;
  if ~isfield(x,'kernParam1')
    egopt.kernParam = eg.kernParam;
  else
    egopt.kernParam1 = reshape(eg.kernParam(2:end),size(x.kernParam1));
    egopt.kernParam = eg.kernParam(1);
  end
end