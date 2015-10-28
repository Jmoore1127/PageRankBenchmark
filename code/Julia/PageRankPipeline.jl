#
include("KronGraph500NoPerm.jl")
include("EdgeFileWrite.jl")
include("EdgeFileRead.jl")

function PageRankPipeline(SCALE,EdgesPerVertex,Nfile);

  Nmax = 2.^SCALE;                                 # Max vertex ID.
  M = EdgesPerVertex .* Nmax;                      # Total number of edges.
  myFiles = [1:Nfile].';                               # Set list of files.
  #
  # Julia parallel verison 
  # figur out later: how to distribute the load
  # myFiles = global_ind(zeros(Nfile,1,map([Np 1],{},0:Np-1)));   # PARALLEL.
  tab = char(9);
  nl = char(10);
  Niter = 20;                                      # Number of PageRank iterations.
  c = 0.15;                                        # PageRank damping factor.

  println("Number of Edges: " * string(M) * ", Maximum Possible Vertex: " * string(Nmax));


  ########################################################
  # Kernel 0: Generate a Graph500 Kronecker graph and save to data files.
  ########################################################
  println("Kernel 0: Generate Graph, Write Edges");
  tic();

    for i in myFiles
      fname = "data/K0/" * string(i) * ".tsv";
      println("  Writing: " * fname);  # Read filename.
      srand(i);                              # Set random seed to be unique for this file.
      # @printf("%.0f \t %.0f\n",SCALE,EdgesPerVertex./Nfile); 
      u, v = KronGraph500NoPerm(SCALE,EdgesPerVertex./Nfile);                    # Generate data.
 
      # edgeStr = @sprintf("%16.16g $tab %16.16g $nl",[u.'; v.'])      # Convert edges to strings.
      # edgeStr = edgeStr(edgeStr ~= ' ');                             # Remove extra spaces.
      # StrFileWrite(edgeStr,fname);                                   # Write string to file.
      # Write edges to file
      EdgeFileWrite(u,v,fname);

    end

  K0time = toq();
  println("K0 Time: " * string(K0time) * ", Edges/sec: " * string(M./K0time));


  ########################################################
  # Kernel 1: Read data, sort data, and save to files.
  ########################################################
  println("Kernel 1: Read, Sort, Write Edges");
  tic();

    # Read in all the files into one array.
    for i in myFiles
      fname = "data/K0/" * string(i) * ".tsv";
      println("  Reading: " * fname);  # Read filename.
      ut,vt = EdgeFileRead(fname);
      # Concatenate to u,v
      if i == 1
         u = ut; v = vt;
      else
         u = hcat(u,ut); v = hcat(v,vt);
      end
    end

    # Sort by start edge.
    # uv = sscanf(edgeStr,'#f');                      # Convert string to numeric data.  # u = uv(1:2:end);                                # Get starting vertices.
    # v = uv(2:2:end);                                # Get ending vertices.

    sortIndex = sortperm(vec(u));                     # Sort starting vertices.
    u = u[sortIndex]
    v = v[sortIndex]

    # Write all the data to files.
    j = 1;                                                         # Initialize file counter.
    for i in myFiles
      jEdgeStart = ((j-1).*(size(u,1)./length(myFiles))+1);        # Compute first edge of file.
      jEdgeEnd = ((j).*(size(u,1)./length(myFiles)));              # Compute last edge of file.
      uu = u[jEdgeStart:jEdgeEnd];                                 # Select start vertices.
      vv = v[jEdgeStart:jEdgeEnd];                                 # Select end vertices.
      fname = "data/K1/" * string(i) * ".tsv";
      println("  Writing: " * fname);                                # Create filename.
      # edgeStr = sprintf(['#16.16g' tab '#16.16g' nl],[uu'; vv']);  # Convert edges to strings.
      # edgeStr = edgeStr(edgeStr ~= ' ');                           # Remove extra spaces.
      # StrFileWrite(edgeStr,fname);                                 # Write string to file.
      EdgeFileWrite(uu,vv,fname);
      j = j + 1;                                                   # Increment file counter.
    end

  K1time = toq();
  println("K1 Time: " * string(K1time) * ", Edges/sec: " * string(M./K1time));


  ########################################################
  # Kernel 2: Read data, filter data.
  ########################################################
  println("Kernel 2: Read, Filter Edges");
  tic();

    # Read in all the files into one array.
    for i in myFiles
      fname = "data/K1/" * string(i) * ".tsv";
      println("  Reading: " * fname);                # Read filename.
      ut,vt = EdgeFileRead(fname);
      if i == 1
         u = ut; v = vt;
      else
         u = hcat(u,ut); v = hcat(v,vt);
      end
    end

    # Construct adjacency matrix.
    # uv = sscanf(edgeStr,'#f');       # Convert string to numeric data.
    # u = uv(1:2:end);                 # Get starting vertices.
    # v = uv(2:2:end);                 # Get ending vertices.
    A = sparse(int(vec(u)),int(vec(v)),1,Nmax,Nmax); # Create adjacency matrix.
    # A = sparse(u,v,1,Nmax,Nmax);     # Create adjacency matrix.

    # Filter and weight the adjacency matrix.
    din = sum(A,1);                    # Compute in degree.
    setindex!(A,0,find(din == maximum(din))) # Eliminate the super-node.
    setindex!(A,0,find(din == 1))        # Eliminate the leaf-node.
    dout = sum(A,2);                   # Compute the out degree.
    i = find(dout);                      # Find vertices with outgoing edges (dout > 0).
    DoutInv = sparse(i,i,1./dout[i],Nmax,Nmax);   # Create diagonal weight matrix.
    A = DoutInv * A;                   # Apply weight matrix.

  K2time = toq();
  println("K2 Time: " * string(K2time) * ", Edges/sec: " * string(M./K2time));


  ########################################################
  # Kernel 3: Compute PageRank.
  ########################################################
  println("Kernel 3: PageRank");
  tic();

    r = rand(Nmax,1);                     # Generate a random starting rank.
    r = r ./ norm(r,1);                   # Normalize
    a = ones(Nmax,1) .* (1-c) ./ Nmax;    # Create damping vector

    for i=1:Niter
      r = A * (r .* c) + a;               # Compute PageRank.
    end

  K3time = toq();
  println("  Sum of PageRank: " * string(sum(r)) );     # Force all computations to occur.
  println("K3 Time: " * string(K3time) * ", Edges/sec: " * string(Niter.*M./K3time));

  return K0time,K1time,K2time,K3time

end

########################################################
# PageRank Pipeline Benchmark
# Architect: Dr. Jeremy Kepner (kepner@ll.mit.edu)
# Julia Translation: Dr. Chansup Byun (cbyun@ll.mit.edu)
# MIT
########################################################
# (c) <2015> Massachusetts Institute of Technology
########################################################


