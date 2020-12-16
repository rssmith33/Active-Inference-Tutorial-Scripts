function Q = spm_MDP_VB_game_tutorial(MDP)
% auxiliary plotting routine for spm_MDP_VB - multiple trials
% FORMAT Q = spm_MDP_VB_game(MDP)
%
% MDP.P(M,T)      - probability of emitting action 1,...,M at time 1,...,T
% MDP.Q(N,T)      - an array of conditional (posterior) expectations over
%                   N hidden states and time 1,...,T
% MDP.X           - and Bayesian model averages over policies
% MDP.R           - conditional expectations over policies
% MDP.O(O,T)      - a sparse matrix encoding outcomes at time 1,...,T
% MDP.S(N,T)      - a sparse matrix encoding states at time 1,...,T
% MDP.U(M,T)      - a sparse matrix encoding action at time 1,...,T
% MDP.W(1,T)      - posterior expectations of precision
%
% MDP.un  = un    - simulated neuronal encoding of hidden states
% MDP.xn  = Xn    - simulated neuronal encoding of policies
% MDP.wn  = wn    - simulated neuronal encoding of precision
% MDP.da  = dn    - simulated dopamine responses (deconvolved)
% MDP.rt  = rt    - simulated dopamine responses (deconvolved)
%
% returns summary of performance:
%
%     Q.X  = x    - expected hidden states
%     Q.R  = u    - final policy expectations
%     Q.S  = s    - initial hidden states
%     Q.O  = o    - final outcomes
%     Q.p  = p    - performance
%     Q.q  = q    - reaction times
%
% please see spm_MDP_VB
%__________________________________________________________________________
% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_VB_game.m 7307 2018-05-08 09:44:04Z karl $

% numbers of transitions, policies and states
%--------------------------------------------------------------------------
if iscell(MDP(1).X)
    Nf = numel(MDP(1).B);                 % number of hidden state factors
    Ng = numel(MDP(1).A);                 % number of outcome factors
else
    Nf = 1;
    Ng = 1;
end

% graphics
%==========================================================================
Nt    = length(MDP);               % number of trials
Ne    = size(MDP(1).V,1) + 1;      % number of epochs per trial
Np    = size(MDP(1).V,2) + 1;      % number of policies
for i = 1:Nt
    
    % assemble expectations of hidden states and outcomes
    %----------------------------------------------------------------------
    for j = 1:Ne
        for k = 1:Ne
            for f = 1:Nf
                try
                    x{f}{i,1}{k,j} = gradient(MDP(i).xn{f}(:,:,j,k)')';
                catch
                    x{f}{i,1}{k,j} = gradient(MDP(i).xn(:,:,j,k)')';
                end
            end
        end
    end
    s(:,i) = MDP(i).s(:,2);
    o(:,i) = MDP(i).o(2,:)';
    act_prob(:,i) = MDP(i).P(:,:,1)';
    act(:,i) = MDP(i).u(2,1);
    w(:,i) = mean(MDP(i).dn,2);
    
    
    % assemble context learning
    %----------------------------------------------------------------------
    for f = 1:Nf
        try
            try
                D = MDP(i).d{f};
            catch
                D = MDP(i).D{f};
            end
        catch
            try
                D = MDP(i).d;
            catch
                D = MDP(i).D;
            end
        end
        d{f}(:,i) = D/sum(D);
    end
    
    % assemble performance
    %----------------------------------------------------------------------
    p(i)  = 0;
    for g = 1:Ng
        try
            U = spm_softmax(MDP(i).C{g});
        catch
            U = spm_softmax(MDP(i).C);
        end
        for t = 1:Ne
            p(i) = p(i) + log(U(MDP(i).o(g,t),t))/Ne;
        end
    end
    q(i)   = sum(MDP(i).rt(2:end));
    
end

% assemble output structure if required
%--------------------------------------------------------------------------
if nargout
    Q.X  = x;            % expected hidden states
    Q.R  = act_prob;     % final policy expectations
    Q.S  = s;            % inital hidden states
    Q.O  = o;            % final outcomes
    Q.p  = p;            % performance
    Q.q  = q;            % reaction times
    return
end


% Initial states and expected policies (habit in red)
%--------------------------------------------------------------------------
col   = {'r.','g.','b.','c.','m.','k.'};
t     = 1:Nt;
subplot(5,1,1)
if Nt < 64
    MarkerSize = 24;
else
    MarkerSize = 16;
end

image(64*(1 - act_prob)),  hold on

plot(act,col{3},'MarkerSize',MarkerSize)

try
    plot(Np*(1 - act_prob(Np,:)),'r')
end
try
    E = spm_softmax(spm_cat({MDP.e}));
    plot(Np*(1 - E(end,:)),'r:')
end
title('Action selection and action probabilities')
xlabel('Trial'),ylabel('Action'), hold off
yticklabels({'Start','Hint','Choose Left','Choose Right'})
% Performance
%--------------------------------------------------------------------------

subplot(5,1,2), bar(p,'k'),   hold on

for i = 1:size(o,2)
%     j(i,1) = max(o(:,i));
    if MDP(i).o(3,2) == 2
        j(i,1) = MDP(i).o(2,3)-1;
    else
        j(i,1) = MDP(i).o(2,2)-1;
    end
    if j(i,1) == 1
        jj(i,1) = 1;
    else
        jj(i,1) = -2;
    end
end



plot((j),col{2},'MarkerSize',MarkerSize);
plot((jj),col{6},'MarkerSize',MarkerSize);


title('Win/Loss and Free energies')
ylabel('Value and Win/Loss'), spm_axis tight, hold off, box off
set(gca,'YTick',[-4:1:3])
yticklabels({'','','','Free Energy','','Loss','Win'})

% Initial states (context)
%--------------------------------------------------------------------------
subplot(5,1,3)
col   = {'r','b','g','c','m','k','r','b','g','c','m','k'};
for f = 1:Nf
    if Nf > 1
        plot(spm_cat(x{f}),col{f}), hold on
    else
        plot(spm_cat(x{f}))
    end
end
title('State estimation (ERPs)'), ylabel('Response'), 
spm_axis tight, hold off, box off

% Precision (dopamine)
%--------------------------------------------------------------------------
subplot(5,1,4)
w   = spm_vec(w);
if Nt > 8
    fill([1 1:length(w) length(w)],[0; w.*(w > 0); 0],'k'), hold on
    fill([1 1:length(w) length(w)],[0; w.*(w < 0); 0],'k'), hold off
else
    bar(w,1.1,'k')
end
title('Precision (dopamine)')
ylabel('Precision','FontSize',12), spm_axis tight, box off
YLim = get(gca,'YLim'); YLim(1) = 0; set(gca,'YLim',YLim);
set(gca,'XTickLabel',{});

% learning - D
%--------------------------------------------------------------------------
for f = 1
    subplot(5*Nf,1,Nf*4 + f), image(64*(1 - d{f}))
    if f < 2
        title('Context Learning')
    end
    set(gca,'XTick',1:Nt);
%     if f < Nf
%         set(gca,'XTickLabel',{});
%     end
%     set(gca,'YTick',1);
%     try
%         set(gca,'YTickLabel',MDP(1).label.factor{f});
%     end
%     try
%         set(gca,'YTickLabel',MDP(1).Bname{f});
%     end
    
    yticklabels({'Left-Win','Right-Win'})
    
end
