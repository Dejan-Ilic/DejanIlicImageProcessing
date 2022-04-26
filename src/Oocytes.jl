using ImageFeatures

function read_oocyte_stack(main_folder::String, h=800, w=800, z=11)
	stack = Array{Float64, 3}(undef, h, w, z)
	i = 0;

	for F=-75:15:75
		i += 1
		path = joinpath(main_folder, "F$F")
		imgname = readdir(path)
		if length(imgname) < 1
			println("error on F$(F), no img")
			continue
		end

		stack[:,:,i] .= load(joinpath(path,imgname[1]))
	end

	return stack
end

function read_oocyte_stack(n::Int, OOCYTE_FOLDER, OOCYTE_LIST)
	if n < 1 || n > length(OOCYTE_LIST)
		throw(DomainError(n, "Out of range: n should be between 1 and $(length(OOCYTE_LIST))"))
	else
        path = joinpath(OOCYTE_FOLDER,OOCYTE_LIST[n])
		read_oocyte_stack(path)
	end
end

#RADIUS DETERMINED EMPIRICALLY
function cell_circle_detection(img, p1=98, p2=80, MIN_CELL_RADIUS = 140, MAX_CELL_RADIUS = 210)
	img_edges = canny(img, (Percentile(p1), Percentile(p2)));
	dx, dy=imgradients(img, KernelFactors.ando5);
	img_phase = phase(dx, dy);

	centers, radii = hough_circle_gradient(img_edges, img_phase, MIN_CELL_RADIUS:MAX_CELL_RADIUS);

	return centers, radii
end

function detect_and_mark_circles(img, p1, p2, m, M)
	H,W = size(img)
	centers, radii = cell_circle_detection(img, p1, p2, m, M)
	colimg = RGB.(img)

	red = RGB(1,0,1)
	smartstep = 360/(2*π*M)/2
	for c in eachindex(centers)
		colimg[centers[c]] = red
		for θ = 0:smartstep:360
			i = round(Int, radii[c] * sin(θ) + centers[c][1] + 0.5)
			j = round(Int, radii[c] * cos(θ) + centers[c][2] + 0.5)

			i = max(1, min(H, i))
			j = max(1, min(W, j))

			colimg[i,j] = red
		end
	end

	colimg
end
