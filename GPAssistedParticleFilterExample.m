clear; close all;

str_traj = {'8cur','lissa','lima'};
n_t = 1000;
n_traj = 2; % specify trajectory type from str_traj = {'8cur','lissa','lima'};
n_sigma = 0.1;

initial_seed = RandStream.create('mrg32k3a','Seed','Shuffle').Seed;
all_seeds = initial_seed:initial_seed;

% anchor positions
rec_width = 30;
rec_height = 30;
n_anchor = 3;
pos_anchor = zeros(2,n_anchor);
pos_anchor(:,1) = [rec_width/2;rec_height/2 + sqrt(3)*rec_width/8];
pos_anchor(:,2) = [rec_width/4;rec_height/2 - sqrt(3)*rec_width/8];
pos_anchor(:,3) = [3*rec_width/4;rec_height/2 - sqrt(3)*rec_width/8];

center = pos_anchor(:,1)+[8;-18];
traj1 = getlemniscateOfBernoulli(center,12,n_t); 
traj2 = getLissajous(pi/2,3,4,rec_width,rec_height,n_t);
traj3 = getLimacon(rec_width,rec_height,n_t);
fh_trajv = {traj1,traj2,traj3};

load('xe_hvM.mat');
load('xe_SE.mat');
load('xe_vM.mat');
load('traininginputs.mat');
load('traininginputs_unit.mat');
load('observations.mat');
load('noise_dist.mat');

% generate trajectory
pos_tag_te = fh_trajv{n_traj};

mean_noise = zeros(2,1);
cov_noise = diag([0.4,0.4].^2);
dist_process_noise = GaussianDist(mean_noise,cov_noise);
cov_noise = diag([1,1]);
prior_noise = GaussianDist(mean_noise,cov_noise);

rng(all_seeds);
rng('shuffle');

[rng_meas_te,rng_gt_te] = getRangeMeasurements(pos_tag_te,pos_anchor,n_sigma);
aoa_te = getAoA(pos_tag_te,pos_anchor);
n_te = size(aoa_te,2);
% particle filter
n_particles = 100;

fh_lkh_HvM = @(x,y) likelihoodGP(x,training_points_unit,obs,pos_anchor,xeopt_hvM,y,{'hypertoroidalvMKernel'});
pos_est_HvM = pf(rng_meas_te,dist_process_noise,prior_noise,fh_lkh_HvM,n_particles,pos_tag_te(:,1));

fh_lkh_SE = @(x,y) likelihoodGP(x,training_points,obs,pos_anchor,xeopt_SE,y,{'SEKernel'});
pos_est_SE = pf(rng_meas_te,dist_process_noise,prior_noise,fh_lkh_SE,n_particles,pos_tag_te(:,1));  

fh_lkh_vM = @(x,y) likelihoodGP(x,training_points,obs,pos_anchor,xeopt_vM,y,{'vMKernel'});
pos_est_vM = pf(rng_meas_te,dist_process_noise,prior_noise,fh_lkh_vM,n_particles,pos_tag_te(:,1));  

fh_lkh_gauss = @(x,y) likelihoodGaussian(x,pos_anchor,y,dist_rng_noise);
pos_est_gauss = pf(rng_meas_te,dist_process_noise,prior_noise,fh_lkh_gauss,n_particles,pos_tag_te(:,1));

plotTraj(pos_anchor,pos_tag_te,pos_est_HvM,4);
plotTraj(pos_anchor,pos_tag_te,pos_est_SE,2);
plotTraj(pos_anchor,pos_tag_te,pos_est_vM,2);
plotTraj(pos_anchor,pos_tag_te,pos_est_gauss,2);

% plot error distribution
test_input = [cos(aoa_te);sin(aoa_te)];
n_tst = size(test_input,2);
[m_est_HvM,covar_HvM] = estMOGP(1,2*n_anchor,n_anchor,xeopt_hvM,{'hypertoroidalvMKernel'},training_points_unit,obs,test_input);  
m_est_grid_HvM = reshape(m_est_HvM,n_tst,n_anchor)';
sigma_HvM = reshape(sqrt(diag(covar_HvM)),n_tst,n_anchor)';

[m_est_SE,covar_SE] = estMOGP(n_anchor,1,n_anchor,xeopt_SE,{'SEKernel'},training_points,obs,aoa_te);  
m_est_grid_SE = reshape(m_est_SE,n_tst,n_anchor)';
sigma_SE = reshape(sqrt(diag(covar_SE)),n_tst,n_anchor)';

[m_est_vM,covar_vM] = estMOGP(n_anchor,1,n_anchor,xeopt_vM,{'vMKernel'},training_points,obs,aoa_te);  
m_est_grid_vM = reshape(m_est_vM,n_tst,n_anchor)';
sigma_vM = reshape(sqrt(diag(covar_vM)),n_tst,n_anchor)';

% figure('Position',get(0, 'Screensize'))
err_te = rng_meas_te - rng_gt_te;
idx_end = 1000;
idx_te = 1:idx_end;
lw = 2;
err_HvM = m_est_grid_HvM - rng_gt_te;
lb_HvM = err_HvM - 2*sigma_HvM;
ub_HvM = err_HvM + 2*sigma_HvM;

err_SE = m_est_grid_SE - rng_gt_te;
lb_SE = err_SE - 2*sigma_SE;
ub_SE = err_SE + 2*sigma_SE;
err_vM = m_est_grid_vM - rng_gt_te;
lb_vM = err_vM - 2*sigma_vM;
ub_vM = err_vM + 2*sigma_vM;

err_gauss = dist_rng_noise.mu.*ones(1,idx_end);
sigma_gauss = sqrt(diag(dist_rng_noise.C)).*ones(1,idx_end);
lb_gauss = err_gauss - 2*sigma_gauss;
ub_gauss = err_gauss + 2*sigma_gauss;

for i = 1:n_anchor
  plotErrorDist(idx_te,lb_HvM(i,idx_te),ub_HvM(i,idx_te),err_te(i,idx_te),err_HvM(i,idx_te),lw);
  plotErrorDist(idx_te,lb_SE(i,idx_te),ub_SE(i,idx_te),err_te(i,idx_te),err_SE(i,idx_te),lw);
  plotErrorDist(idx_te,lb_vM(i,idx_te),ub_vM(i,idx_te),err_te(i,idx_te),err_vM(i,idx_te),lw);
  plotErrorDist(idx_te,lb_gauss(i,idx_te),ub_gauss(i,idx_te),err_te(i,idx_te),err_gauss(i,idx_te),lw);
end

function plotTraj(pos_anchor,pos_tag_te,pos_est_vM,lw)
  figure;
  scatter(pos_anchor(1,:),pos_anchor(2,:),80,'filled','MarkerFaceColor','#0a481e','MarkerFaceAlpha',0.8)
  hold on 
  plot(pos_tag_te(1,:),pos_tag_te(2,:),'r','LineWidth',lw);  
  plot(pos_est_vM(1,:),pos_est_vM(2,:),'Color','b','LineWidth',2);
  xlabel('x [m]');
  ylabel('y [m]');
  ax = gca; 
  ax.FontSize = 22;   
  set(gca,'TickLabelInterpreter','latex')
  grid on
  box on
  set(gca,'GridLineStyle','--')
  xlim([0,30])
  ylim([0,30])  
end

function pos = getlemniscateOfBernoulli(center,a,n_t)
  t = linspace(0,2*pi,n_t);
  cos_ct = cos(t);
  sin_ct = sin(t);
  plus_sin2_t = 1 + sin_ct.*sin_ct;
  
  x = a*sin_ct.*cos_ct ./ plus_sin2_t;
  y = a.*cos_ct ./ plus_sin2_t;
  pos = [x + center(1);
	       y + center(2)];
  theta = pi/6;
  R = [cos(theta),-sin(theta);
       sin(theta),cos(theta)];
  pos = R*pos;
end

function pos = getLissajous(delta,a,b,rec_width,rec_height,n_t)
  t = linspace(0,2*pi,n_t);
  x = 0.4*rec_width*sin(a*t + delta);
  y = 0.4*rec_width*sin(b*t);
  pos = [x + rec_width/2;
         y + rec_height/2];
end

function pos = getLimacon(rec_width,rec_height,n_t)
  phi = linspace(0,2*pi,n_t);
  r = 0.28*rec_width*(1 + 2*cos(phi));
  [x,y] = pol2cart(phi,r);
  pos = [x;y];  
  theta = -pi/2;
  R = [cos(theta),-sin(theta);
       sin(theta),cos(theta)];
  pos = R*pos;  
  pos = pos + [rec_width/2;rec_height/2+11];    
end

function aoa = getAoA(pos_tag,pos_anchor)
  n_anchor = size(pos_anchor,2);
  n_smp = size(pos_tag,2);  
  aoa = zeros(n_anchor,n_smp);
  for i = 1:n_anchor
    aoa(i,:) = atan2(pos_anchor(2,i) - pos_tag(2,:),pos_anchor(1,i) - pos_tag(1,:));
  end
end

function [rng_meas,rng_gt] = getRangeMeasurements(pos_tag,pos_anchor,n_sigma)
  % Returns:
  %   input (2 x m matrix)
  %     each column represents represents one of the test inputs
  %   output (3 x m matrix)
  %     groundtruth at test points   
  n_anchor = size(pos_anchor,2);
  n_smp = size(pos_tag,2);
  rng_gt = zeros(n_anchor,n_smp);
  for j = 1:n_anchor
    rng_gt(j,:) = sqrt(sum((pos_tag - pos_anchor(:,j)).^2));
  end

  rng_meas = zeros(n_anchor,n_smp);
  gauss = GaussianDist(zeros(n_smp,1),diag(n_sigma^2*ones(1,n_smp).^2));
  for i = 1:n_anchor
    noise = gauss.sample(1)';
    rng_meas(i,:) = rng_gt(i,:) + (0.05*rng_gt(i,:)) + noise;
  end
end

function [pdf,valid] = likelihoodGP(pos_est,training_inputs,obs,pos_anchor,xeopt,rng_meas,kernel_type)
  % compute test AOA
  n_tst = size(pos_est,2);
  n_anchor = size(pos_anchor,2);
  aoa = zeros(n_anchor,n_tst);
  for i = 1:n_anchor
    aoa(i,:) = atan2(pos_anchor(2,i) - pos_est(2,:),pos_anchor(1,i) - pos_est(1,:));
  end
  % compute distriburion at test input
  if strcmp(kernel_type,'hypertoroidalvMKernel')
    test_input = [cos(aoa);sin(aoa)];
    [m_est,covar] = estMOGP(1,2*n_anchor,n_anchor,xeopt,kernel_type,training_inputs,obs,test_input);
  else
    test_input = aoa;
    [m_est,covar] = estMOGP(n_anchor,1,n_anchor,xeopt,kernel_type,training_inputs,obs,test_input);
  end
  m_est_grid = reshape(m_est,n_tst,n_anchor)';
  err = rng_meas - m_est_grid;
  if any(abs(err) > 3)
    valid = 0;
    pdf = 0;
    return
  end     
  % compute pdf
  pdf = zeros(1,n_tst);
  for i = 1:n_tst
    cov = covar(i:n_tst:end,i:n_tst:end);
    cov = 0.5*(cov + cov');
    gaussd = GaussianDist(zeros(n_anchor,1),cov);
    pdf(i) = gaussd.pdf(err(:,i));
  end
  valid = 1;
  if (~sum(pdf)>0) 
    valid = 0;
  end
end

function [pdf,valid] = likelihoodGaussian(pos_est,pos_anchor,rng_meas,dist_rng_noise)
  % compute predicted range
  n_tst = size(pos_est,2);
  n_anchor = size(pos_anchor,2);
  rng_pred = zeros(n_anchor,n_tst);
  for i = 1:n_anchor
    rng_pred(i,:) = sqrt(sum((pos_est - pos_anchor(:,i)).^2,1));
  end
  % compute pdf
  err = rng_meas - rng_pred;
  if any(abs(err) > 3)
    valid = 0;
    pdf = 0;
    return
  end
  pdf = dist_rng_noise.pdf(err);
  valid = 1;
  if (~sum(pdf)>0) 
    valid = 0;
  end  
end