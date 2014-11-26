import org.la4j.vector.functor.VectorFunction;

public class LogSig implements VectorFunction {

	public double evaluate(int arg0, double arg1) {
		return 1/ (1 + Math.exp(- arg1));
	}

}
