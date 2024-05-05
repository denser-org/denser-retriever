import { useRouter } from "next/router";
import { Background } from "./background";

const pathsWithoutFooterWidgets = ["/imprint", "/blog"];

export const MainContainer = (props) => {
  const router = useRouter();

  return (
    <>
      {props.children}
      <Background />
    </>
  );
};
