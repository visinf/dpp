require 'gnuplot'

local M = {}

local function readLog(filename)
	local file = io.open(filename)
	local count = 0
	file:read()
	local current = file:seek() 
	for line in file:lines() do
		count = count + 1
	end
	file:seek("set", current)

	local trainError = torch.FloatTensor(count)
	local valError = torch.FloatTensor(count)
	for i=1,count do
		valError[i] = file:read('*n') -- val error is logged first
		trainError[i] = file:read('*n')
	end 
		
	file:close()
	return trainError, valError
end

function M.plotLogPDF(filename)  
	local trainError, valError = readLog(filename)
	gnuplot.pdffigure(filename:sub(1, filename:len()-4) .. '.pdf')
	gnuplot.plot({'Train Error', trainError, '-'}, {'Val Error', valError, '-'})
	gnuplot.plotflush()
	gnuplot.xlabel('Epoch')
	gnuplot.ylabel('Error')
	gnuplot.axis({1, "", 0, 100})
	gnuplot.grid(true)																																																																																																																																					
	gnuplot.title('Model Convergence Curves')
end

return M
