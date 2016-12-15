close all 
clc,clear
e_t = load('error_train.mat');
e_v = load('error_valid.mat');
e_ts = load('error_train_s.mat');
e_vs = load('error_valid_s.mat');
figure
plot(e_t.err_train,'LineWidth',2)
hold on
plot(e_ts.err_train,'LineWidth',2)
hold off
l = legend('fixed prior variance','alternately updating variance with U,P');
set(l,'FontSize',12)

xlabel('iteration number','FontSize',14)
ylabel('value of objective function','FontSize',14)

figure()
plot(e_v.err_valid,'LineWidth',2)
hold on
plot(e_vs.err_valid,'LineWidth',2)
hold off
l=legend('fixed prior variance','alternately updating variance with U,P');
set(l,'FontSize',12)

xlabel('iteration number','FontSize',14)
ylabel('RMSE','FontSize',14)