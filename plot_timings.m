function plot_timings()
close all;
% reading C output

threads_new=[1,4,8,16];
threads_old=[1,4,8,16];
threads_cpu=[1,4,8,16];

X = cell(length(threads_new),1);
Y = cell(length(threads_old),1);
Z = cell(length(threads_cpu),1);


for i=1:length(threads_new)
    filename = strcat(['./tests/new_version_',num2str(threads_new(i)),'.txt']);
    fileID = fopen(filename,'r');
    formatSpec = '%lf';
    A = fscanf(fileID,formatSpec);
    X{i} = reshape(A,[7,length(A)/7]);
end

for i=1:length(threads_old)
    filename = strcat(['./tests/old_version_',num2str(threads_old(i)),'.txt']);
    fileID = fopen(filename,'r');
    formatSpec = '%lf';
    A = fscanf(fileID,formatSpec);
    Y{i} = reshape(A,[7,length(A)/7]);
end

for i=1:length(threads_cpu)
    filename = strcat(['./tests/cpu_version_',num2str(threads_cpu(i)),'.txt']);
    fileID = fopen(filename,'r');
    formatSpec = '%lf';
    A = fscanf(fileID,formatSpec);
    Z{i} = reshape(A,[7,length(A)/7]);
end

dim = [2,2];

%% Time per iteration

fig = figure('visible','off','Renderer', 'painters');
fig.Position(3:4)=[1000,800];
sgtitle('Average Wall Clock Time per Iteration','fontsize',20)
for i=1:length(threads_new)
    subplot(dim(1),dim(2),i)
    new_time = X{i}(7,:)./X{i}(1,:);
    plot(X{i}(5,:),new_time,'*',...
        'displayname','New GPU Version')
    hold on
    old_time = Y{i}(7,:)./Y{i}(1,:);
    plot(Y{i}(5,:),old_time,'*',...
        'displayname','Original GPU Version')
    cpu_time = Z{i}(7,:)./Z{i}(1,:);
    plot(Z{i}(5,:),cpu_time,'*',...
        'displayname','Original CPU Version')
    hold off
    ylabel('$t\:[s]$','interpreter','latex','fontsize',14)
    xlabel('Grid Size $N_x\times N_y \times N_z$','interpreter','latex','fontsize',14)
    legend('location','nw')
    title(strcat([num2str(threads_new(i)),' Threads']),...
        'interpreter','latex','fontsize',16)
    grid()
end

%saveas(fig,'./figures/running_time.png')
z=getframe(fig);
imwrite(z.cdata,'./figures/running_time.png')

%% Time per Gonjugate Gradient Iteration

fig = figure('visible','off','Renderer', 'painters');
fig.Position(3:4)=[1000,800];
sgtitle('Average Wall Clock Time per Conjugate Gradient Iteration','fontsize',20)
for i=1:length(threads_new)
    subplot(dim(1),dim(2),i)
    new_time = X{i}(7,:)./X{i}(6,:);
    plot(X{i}(5,:),new_time,'*',...
        'displayname','New GPU Version')
    hold on
    old_time = Y{i}(7,:)./Y{i}(6,:);
    plot(Y{i}(5,:),old_time,'*',...
        'displayname','Original GPU Version')
    cpu_time = Z{i}(7,:)./Z{i}(6,:);
    plot(Z{i}(5,:),cpu_time,'*',...
        'displayname','Original CPU Version')
    hold off
    ylabel('$t\:[s]$','interpreter','latex','fontsize',14)
    xlabel('Grid Size $N_x\times N_y \times N_z$','interpreter','latex','fontsize',14)
    legend('location','nw')
    title(strcat([num2str(threads_new(i)),' Threads']),...
        'interpreter','latex','fontsize',16)
    grid()
end

z=getframe(fig);
imwrite(z.cdata,'./figures/cg_running_time.png')

%% Time per Gonjugate Gradient Iteration

fig = figure('visible','off','Renderer', 'painters');
fig.Position(3:4)=[500,400];
new_time = X{1}(7,:)./X{1}(1,:);
old_time = Y{1}(7,:)./Y{1}(1,:);
plot(X{1}(5,:),old_time./new_time,'*',...
        'displayname','1 Thread - Old GPU')
hold on
for i=2:length(threads_new)
    new_time = X{i}(7,:)./X{i}(1,:);
    old_time = Y{i}(7,:)./Y{i}(1,:);
    plot(X{i}(5,:),old_time./new_time,'*',...
        'displayname',strcat([num2str(threads_new(i)),' Threads']))
end
hold off
ylabel('$\frac{t_{original}}{t_{new}}$','interpreter','latex','fontsize',16)
xlabel('Grid Size $N_x\times N_y \times N_z$','interpreter','latex','fontsize',14)
legend('location','ne')
grid()
title('Speed-up','fontsize',18)

z=getframe(fig);
imwrite(z.cdata,'./figures/speedup.png')
end
% function plot_timings()
% close all;
% % reading C output
% 
% threads_new=[1,4,8,16];
% threads_old=[1,4,8,16];
% 
% X = cell(length(threads_new),1);
% Y = cell(length(threads_old),1);
% 
% for i=1:length(threads_new)
%     filename = strcat(['./tests/new_version_',num2str(threads_new(i)),'.txt']);
%     fileID = fopen(filename,'r');
%     formatSpec = '%lf';
%     A = fscanf(fileID,formatSpec);
%     X{i} = reshape(A,[7,length(A)/7]);
% end
% 
% for i=1:length(threads_old)
%     filename = strcat(['./tests/old_version_',num2str(threads_old(i)),'.txt']);
%     fileID = fopen(filename,'r');
%     formatSpec = '%lf';
%     A = fscanf(fileID,formatSpec);
%     Y{i} = reshape(A,[7,length(A)/7]);
% end
% 
% dim = [1,2];
% % 'visible','off',
% fig = figure('Renderer', 'painters', 'Position', [10 10 1000 330]);
% sgtitle('Average Wall Clock Time per Iteration','fontsize',20)
% subplot(dim(1),dim(2),1)
% plot(X{1}(5,:),X{1}(7,:)./X{1}(1,:),'*',...
%     'displayname',strcat([num2str(threads_new(1)),' threads']))
% hold on
% for i=2:length(threads_new)
%     plot(X{i}(5,:),X{i}(7,:)./X{i}(1,:),'*',...
%     'displayname',strcat([num2str(threads_new(i)),' threads']))
% end
% hold off
% ylim([0,20])
% ylabel('$t\:[s]$','interpreter','latex','fontsize',16)
% xlabel('Grid Size','interpreter','latex','fontsize',16)
% legend('location','nw')
% title('New GPU Version','interpreter','latex','fontsize',18)
% 
% subplot(dim(1),dim(2),2)
% plot(Y{1}(5,:),Y{1}(7,:)./Y{1}(1,:),'*',...
%     'displayname',strcat([num2str(threads_old(1)),' threads']))
% hold on
% for i=2:length(threads_old)
%     plot(Y{i}(5,:),Y{i}(7,:)./Y{i}(1,:),'*',...
%     'displayname',strcat([num2str(threads_old(i)),' threads']))
% end
% hold off
% %ylim([0,20])
% ylabel('$t\:[s]$','interpreter','latex','fontsize',16)
% xlabel('Grid Size','interpreter','latex','fontsize',16)
% legend('location','nw')
% title('Old GPU Version','interpreter','latex','fontsize',18)
% %saveas(fig,'./figures/running_time.png')
% end