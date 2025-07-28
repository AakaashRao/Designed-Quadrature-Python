% Calculate draws of standardized random terms for random coefficients and save if specified
% Written by Kenneth Train, August 6, 2006.

% Array of draws has dimension NDRAWSxNPxNV.


function draw2=makedraws(NV, NP, DRAWTYPE, NDRAWS)

dr=zeros(NDRAWS,NP,NV);


if DRAWTYPE == 2 || DRAWTYPE == 3   % Halton draws
    h=primes(100);                                % Must create for all people together
    k=1;
    while size(h,2) < NV
        h=primes(k.*100);
        k=k+1;
    end
    h=h(1,1:NV);
    for j=1:NV
        hh=h(1,j);
        draws=[0];
        test=0;
        b=1;
        while test == 0
            drawsold=draws;
            for m=1:(hh-1)
                dd=m./(hh.^b);
                draws=[draws ; drawsold + dd];
                test=size(draws,1) >= ((NP*NDRAWS) + 10);
                if test == 1
                    break
                end
            end
            b=b+1;
        end
        draws=draws(11:(NP*NDRAWS)+10,1);
        if DRAWTYPE == 3
            draws=draws+rand(1,1);               %Shift: one shift for entire sequence
            draws=draws-floor(draws);
            draws=reshape(draws,NDRAWS,NP);
            for n=1:NP                           %Shuffle for each person separately
                rr=rand(NDRAWS,1);
                [~, rrid]=sort(rr);
                draws(:,n)=draws(rrid,n);
            end
            draws=reshape(draws,NP*NDRAWS,1);
        end
        draws=-sqrt(2).*erfcinv(2.*draws);  %Take inverse cum normal
        dr(:,:,j)=reshape(draws,NDRAWS,NP);
    end
end

if DRAWTYPE == 4   % MLHS
    h=0:(NDRAWS-1);
    h=h'./NDRAWS;
    for j=1:NV
        for n=1:NP
            draws=h+rand(1,1)./NDRAWS;    %Shift: Different shift for each person
            rr=rand(NDRAWS,1);
            [~, rrid]=sort(rr);
            draws=draws(rrid,1);          %Shuffle
            draws=-sqrt(2).*erfcinv(2.*draws);  %Take inverse cum normal
            dr(:,n,j)=draws;
        end
    end
end

draw1 = permute(dr,[3,1,2]);
draw2=squeeze(reshape(draw1,NV,1,NP*NDRAWS));