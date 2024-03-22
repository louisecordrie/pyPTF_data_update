function outVal = scalinglaw_Tonini_Murotani(typeScal,M)
%
% From email Roberto Tonini 29/04/2019 and follow up
% def fault_area_murotani(m):
%     a = -3.806
%     b = 1.000
%     area = np.power(10, (a + b*m))
%     return float(area)
% 
% def fault_length_murotani(m):
%     area = fault_area_murotani(m)
%     return float(np.sqrt(area*2.5))
% 
% def fault_width_murotani(m):
%     return float(fault_length_murotani(m)/2.5)

    a =-3.806;
    b =1.000;         
    areaTmp = 10.^(a+b*M); 
    if strcmp(typeScal,'M2W')
      outVal = sqrt(2.5*areaTmp)/2.5;
    elseif strcmp(typeScal,'M2L') 
      outVal = sqrt(2.5*areaTmp);
    else
      disp(['Type ' typeScal ' not recongnized!!']);
    end  

end




